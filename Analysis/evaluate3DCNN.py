# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright(c) 2019 Aalborg University
# Joakim Bruslund Haurum, May 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import argparse
import numpy as np
import os
import csv
import utils
from metrics import calculate_AUC, calculate_classification_metrics, calculate_classification_metrics_full_dataset


## NOTE: Using the MCC, PR Curve and ROC curves as metrics, as per tip 8 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5721660/pdf/13040_2017_Article_155.pdf


def analyze_3DCNN_predictions(predictions, labels, offset, FPM, thresholds = [0.5]):
    """
    Takes a set of predictions from a file, along with the label information, and potential thresholds

    Two sets of predictions are analyzed:
        Per frame: Each prediction corresponds to a set of 16 frames. Here we take each prediction and compares to the label for the minute the first frame in the sequence corresponds to
        Per minute: All predictions from sequences starting during the same minute are averaged and compared to the label
    
    predictions: pandas dataframe containing all predictions to be analyzed. has the following headers:
                    [pred, logsoftmax, filename, frameStart]
    labels: list of per minute binary labels for the file where the predictions dataframe is from. 1 = rain, 0 = no rain. 
    offset: the amount of frames left of the first minute in the video
    FPM: The amount of frames in a minute in the video
    thresholds: a list of thresholds for the per minute predictions


    Output:
        per_frame_dict: A dict containing the binary classification metrics for per frame predictions
        per_minute_dict: A dict containing the binary classification metrics for per minute predictions
        per_frame_labels: A list of the binary labels used for the per frame metrics
        per_minute_labels: A list of the binary labels used for the per minute metrics

    """

    per_frame_labels = []
    per_frame_minutes = []

    # Go throug each prediction and save label and minute of the prediction
    for _, pred in predictions.iterrows():
        label, minute = utils.get_frame_label(labels, offset, FPM, pred[3])
        per_frame_labels.append(label)
        per_frame_minutes.append(minute)
    per_frame_labels = np.asarray(per_frame_labels).astype(np.bool)    
    
    ## SAMPLED PER FRAME
    per_frame_dict = {"Type Errors": [],
                       "Type Rates": [],
                       "Predictive Values": [],
                       "Accuracy": [],
                       "Precision-Recall": [],
                       "F1-score": [],
                       "Informedness": [],
                       "Markedness": [],
                       "MCC": []}
    per_frame_dict = calculate_classification_metrics(per_frame_dict, np.asarray(predictions["pred"],dtype=np.bool), per_frame_labels)
    
    
    ## SAMPLED PER MINUTE
    per_minute_predictions = []
    per_minute_labels = []
    
    # Go through each minute and take the mean label value of all predictions in that minute
    for minute in set(per_frame_minutes):
        index_list = [ True if x == minute else False for x in per_frame_minutes ]
        per_minute_predictions.append(np.mean(predictions["pred"][index_list]))
        per_minute_labels.append(labels[minute])
    per_minute_predictions = np.asarray(per_minute_predictions)
    per_minute_labels = np.asarray(per_minute_labels).astype(np.bool)
        
    
    per_minute_dict = {"Type Errors": [],
                       "Type Rates": [],
                       "Predictive Values": [],
                       "Accuracy": [],
                       "Precision-Recall": [],
                       "F1-score": [],
                       "Informedness": [],
                       "Markedness": [],
                       "MCC": []}

    # Calculate the different binary metrics over different thresholds
    for threshold in thresholds:
       prediction_threshold = per_minute_predictions >= threshold
       per_minute_dict = calculate_classification_metrics(per_minute_dict, prediction_threshold, per_minute_labels)
    
    TPR = [x[0] for x in per_minute_dict["Type Rates"]]
    FPR = [x[2] for x in per_minute_dict["Type Rates"]]
    Precision = [x[0] for x in per_minute_dict["Precision-Recall"]]
    Recall = [x[1] for x in per_minute_dict["Precision-Recall"]]
    per_minute_dict["AUROC"] = calculate_AUC(FPR, TPR)
    per_minute_dict["AUPR"] = calculate_AUC(Recall, Precision)
    
    return per_frame_dict, per_minute_dict, per_frame_labels, per_minute_labels
    



def evaluate3DCNN(args):
    '''
    Evalaute detections from the 3D CNN rain detection algorithm. 
    Saves:
        Plots of the different metrics
        CSV containing metrics for each input file and accumulated
        text file containing results and additional information

    Input:
        args:
            - labelFile: Path to the label file
            - inputFolder: Path to the folder containing the different detection csv files
            - outputFolder: Path to the folder where the output will be saved
            - filePlots: Whether to save plots for each input file
    '''

    label_file = args["labelFile"]
    main_path = args["inputFolder"]
    output_path = args["outputFolder"]
    plots_per_file = args["filePlots"]

    if "laser" in label_file:
        label_type = "Laser"
    else:
        label_type = "Mechanical"

    # Setup output paths
    main_output_path = os.path.join(output_path, "{}-{}-{}".format(os.path.basename(main_path), label_type, "3DCNN"))
    if not os.path.exists(main_output_path):
        os.makedirs(main_output_path)  

    output_path = os.path.join(main_output_path, "results_collected.csv")    

    # Set threshod values
    thresholds = [x for x in np.linspace(0,1,101)]
    label_dict = utils.load_labels(label_file)

    # Containers for the type errors and counters for different label types
    per_minute_counter = np.zeros((101, 4))
    per_frame_counter = np.zeros((1, 4))
    label_total_per_minute = 0
    label_pos_per_minute = 0
    label_total_per_frame = 0
    label_pos_per_frame = 0
            
    with open(output_path, 'w', newline = "") as csvWriteFile:
        writer = csv.writer(csvWriteFile, delimiter=";")
        
        # Write the headers in the new csv file
        firstrow = []
        firstrow.append("file")
        firstrow.append("Total Frames")
        firstrow.append("Rain Frames")
        firstrow.append("%")
        firstrow.append("TP")
        firstrow.append("TN")
        firstrow.append("FP")
        firstrow.append("FN")
        firstrow.append("Accuracy")
        firstrow.append("F1-Score (TP)")
        firstrow.append("F1-Score (TN)")
        firstrow.append("MCC")
            
        writer.writerow(firstrow)
        
        for subdir in os.listdir(main_path):       
            if os.path.isdir(os.path.join(main_path, subdir)):
                continue
            if os.path.splitext(subdir)[-1] == ".txt":
                continue
            
            cnn_file = os.path.join(main_path, subdir)
            df = pd.read_csv(cnn_file, sep=";")

            filenames = df.filename.unique()
            filenames = sorted(filenames)
            print("Number of files: {}\nFilenames: {}".format(len(filenames), filenames))
            
            for filename in filenames:
                print()
                print(filename)

                ###### LOAD DATA ########
                predictions = df.loc[df['filename'] == filename]
                total_pred = len(predictions)

                ###### LOAD LABELS ######            
                label_filename = filename.replace(".mkv",".mp4")
                label_filename = label_filename.replace("-brick","")
                dict_ind = label_dict[os.path.basename(label_filename)]
                offset = dict_ind["frameOffset"]    # How many frames left of the starting minute e.g. 16:00:45, has 15 seconds left 
                                                    # This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
                                                    # These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:00:01
                FPM = dict_ind["FPM"]  # Frames per minute
                labels = dict_ind["labels"] # List of labels per minute
                frameCount = dict_ind["frameCount"]     

                maxFrameStart = np.max(predictions["frameStart"])+1
                print("Total frames in video:  {}\nLargest frame analyzed: {}\nDifference: {}".format(frameCount, maxFrameStart, frameCount-maxFrameStart))

                ####### ANALYSE DATA #######
                if frameCount < total_pred:
                    print("\tStart frame {} is larger than the amount of frames in the labels {}. Skipping this one\n".format(total_pred, frameCount))
                    continue
                
                per_frame, per_minute, per_frame_labels, per_minute_labels = analyze_3DCNN_predictions(predictions, labels, offset, FPM, thresholds = thresholds)
                per_frame_counter += np.asarray(per_frame["Type Errors"])
                per_minute_counter += np.asarray(per_minute["Type Errors"])
                if plots_per_file:
                    utils.make_metrics_plots(per_frame, thresholds, main_output_path, filename.replace(".mp4",".pdf"))
                
                
                row = []
                row.append(filename)
                row.append(total_pred)
                row.append(np.sum(predictions["pred"]))
                row.append(np.sum(predictions["pred"])/total_pred * 100)
                row.append(per_frame["Type Errors"][0][0])
                row.append(per_frame["Type Errors"][0][1])
                row.append(per_frame["Type Errors"][0][2])
                row.append(per_frame["Type Errors"][0][3])
                row.append(per_frame["Accuracy"][0])
                row.append(per_frame["F1-score"][0][0])
                row.append(per_frame["F1-score"][0][1])
                row.append(per_frame["MCC"][0])
                writer.writerow(row)

                label_total_per_minute += len(per_minute_labels)
                label_pos_per_minute += sum(per_minute_labels)
                label_total_per_frame += len(per_frame_labels)
                label_pos_per_frame += sum(per_frame_labels)
        
            # Calculate all metrics based on the accumlated type errors
            total_per_minute = calculate_classification_metrics_full_dataset(per_minute_counter, thresholds)
            total_per_frame = calculate_classification_metrics_full_dataset(per_frame_counter)

            # Make plots of the different metrics
            utils.make_metrics_plots(total_per_minute, thresholds, main_output_path, "overall_3DCNN.pdf")
            
            # Save results
            with open(os.path.join(main_output_path,'evaluation_information.txt'), 'w') as f: 
                f.write("Metrics per frame: {}\n".format(total_per_frame))
                f.write("{} % Rain labels (Per minute)\n".format(label_pos_per_minute/label_total_per_minute * 100))
                f.write("{} rainy out of {} (Per minute)\n\n".format(label_pos_per_minute, label_total_per_minute))
                f.write("{} % Rain labels (Per frame)\n".format(label_pos_per_frame/label_total_per_frame * 100))
                f.write("{} rainy out of {} (Per frame)\n\n".format(label_pos_per_frame, label_total_per_frame))
                f.write("Label file used: {}\n".format(label_file))
                f.write("Method used: 3DCNN\n")
                f.write("Label type used: {}".format(label_type))
            
            
            row = []
            row.append("Total")
            row.append("")
            row.append("")
            row.append("")
            row.append(total_per_frame["Type Errors"][0][0])
            row.append(total_per_frame["Type Errors"][0][1])
            row.append(total_per_frame["Type Errors"][0][2])
            row.append(total_per_frame["Type Errors"][0][3])
            row.append(total_per_frame["Accuracy"][0])
            row.append(total_per_frame["F1-score"][0][0])
            row.append(total_per_frame["F1-score"][0][1])
            row.append(total_per_frame["MCC"][0])
            writer.writerow(row)




if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes a set of output files from the 3DCNN and compares them to the provided labels file. Calcualtes a plethora of binary classification meassures")
    ap.add_argument("-inputFolder", "--inputFolder", type=str, required = True,
                    help="Path to the main folder holding all the csv files to be evaluated")
    ap.add_argument("-labelFile", "--labelFile", type=str, required = True,
                    help="Path to the label file to be used")
    ap.add_argument("-outputFolder", "--outputFolder", type=str, required = True,
                    help="Path to the folder where the output plots and csv file should be saved")
    ap.add_argument('--filePlots', action="store_true",
                    help='saves metrics plots per input file')
    
    args = vars(ap.parse_args())
    
    evaluate3DCNN(args)