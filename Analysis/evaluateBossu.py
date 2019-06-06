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
import numpy as np
import os
import csv
import utils
import argparse
from metrics import calculate_AUC, calculate_classification_metrics, calculate_classification_metrics_full_dataset

## NOTE: Using the MCC, PR Curve and ROC curves as metrics, as per tip 8 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5721660/pdf/13040_2017_Article_155.pdf


    
def analyze_Bossu_predictions(predictions, labels, offset, FPM, start_frame, thresholds = [0.5]):
    """
    Takes a set of predictions from a file, along with the label information, and potential thresholds

    Two sets of predictions are analyzed:
        Per frame: Each prediction corresponds to a single frame. Here we take each prediction and compares to the label for the minute the frame corresponds to
        Per minute: All predictions from sequences starting during the same minute are averaged and compared to the label
    
    predictions: List of prediction values from the Bossu algorithm. Each element corresponds to a frame
    labels: list of per minute binary labels for the file where the predictions dataframe is from. 1 = rain, 0 = no rain. 
    offset: the amount of frames left of the first minute in the video
    FPM: The amount of frames in a minute in the video
    start_frame: The first frame that a prediction was made
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
    for frameNum in range(start_frame, start_frame+len(predictions)):
        label, minute = utils.get_frame_label(labels, offset, FPM, frameNum)
        per_frame_labels.append(label)
        per_frame_minutes.append(minute)
    per_frame_labels = np.asarray(per_frame_labels).astype(np.bool)  
    per_frame_minutes = np.asarray(per_frame_minutes)

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
    per_frame_dict = calculate_classification_metrics(per_frame_dict, np.asarray(predictions,dtype=np.bool), per_frame_labels)
    
    
    ## SAMPLED PER MINUTE
    per_minute_predictions = []
    per_minute_labels = []

    # Go through each minute and take the mean label value of all predictions in that minute
    for minute in set(per_frame_minutes):
        index_list = [ True if x == minute else False for x in per_frame_minutes ]
        per_minute_predictions.append(np.mean(predictions[index_list]))
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
    


def evaluateBossu(args):
    '''
    Evalaute detections from the Bossu rain detection algorithm. 
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
    main_output_path = os.path.join(output_path, "{}-{}-{}".format(os.path.basename(main_path), label_type, "Bossu"))
    if not os.path.exists(main_output_path):
        os.makedirs(main_output_path)  

    output_path = os.path.join(main_output_path, "results_collected.csv")    

    # Set threshod values
    thresholds = [x for x in np.linspace(0,1,101)]
    label_dict = utils.load_labels(label_file)

    # Containers for the type errors and counters for different label types
    em_per_minute_counter = np.zeros((101, 4))
    kalman_sampled_counter = np.zeros((101, 4))
    em_per_frame_counter = np.zeros((1, 4))
    kalman_per_frame_counter = np.zeros((1, 4))

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
        firstrow.append("EM Rain Frames")
        firstrow.append("Kalman Rain Frames")
        firstrow.append("EM %")
        firstrow.append("Kalman %")
        firstrow.append("TP")
        firstrow.append("TN")
        firstrow.append("FP")
        firstrow.append("FN")
        firstrow.append("Accuracy")
        firstrow.append("F1-Score (TP)")
        firstrow.append("F1-Score (TN)")
        firstrow.append("MCC")
        firstrow.append("Kalman TP")
        firstrow.append("Kalman TN")
        firstrow.append("Kalman FP")
        firstrow.append("Kalman FN")
        firstrow.append("Kalman Accuracy")
        firstrow.append("Kalman F1-Score (TP)")
        firstrow.append("Kalman F1-Score (TN)")
        firstrow.append("Kalman MCC")
            
        writer.writerow(firstrow)
        
        for dirs in os.listdir(main_path):
            
            dir_path = os.path.join(main_path, dirs)
            print("\n{}".format(dirs))
        
            dir_content = os.listdir(dir_path)
            
            settings = [s for s in dir_content if "setting" in s.lower()]
            if len(settings) > 1:
                raise ValueError("more than one settings file present in {}".format(dir_path))
            
            
            for subdir in dir_content:
                if os.path.isdir(os.path.join(dir_path, subdir)):
                    continue
                if os.path.splitext(subdir)[-1] == ".txt":
                    continue
                
                
                ###### LOAD LABELS ######
                filename = subdir.replace(".mkv",".mp4")[:-12]
                filename = filename.replace("-brick","")
                
                print(filename)
                
                dict_ind = label_dict[os.path.basename(filename)]
                offset = dict_ind["frameOffset"]    # How many frames left of the starting minute e.g. 16:00:45, has 15 seconds left 
                                                    # This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
                                                    # These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:00:01
                FPM = dict_ind["FPM"]  # Frames per minute
                labels = dict_ind["labels"] # List of labels per minute
                frameCount = dict_ind["frameCount"]      
                
                
                ###### LOAD BOSSU OUTPUT ######
                # Load the supplied csv file
                csv_file = os.path.join(dir_path, subdir)
                rain_dataframe = pd.read_csv(csv_file, sep=";")
                
                
                ####### ANALYSE DATA #######
                start_frame = rain_dataframe[" Frame#"][0] - 1
                total_frames = len(rain_dataframe[" Frame#"])
                    
                maxFrameStart = np.max(rain_dataframe[" Frame#"])
                print("Total frames in video:  {}\nLargest frame analyzed: {}\nDifference: {}".format(frameCount, maxFrameStart, frameCount-maxFrameStart))  
                if frameCount != (total_frames+start_frame):
                    print("\tSize mismatch between labels {}, and data, {}. Skipping this one\n".format(frameCount, total_frames))
                    continue
                
                em_detected = rain_dataframe["EM Rain Detected"]
                kalman_detected = rain_dataframe["Kalman Rain Detected"]
                
                ## Raw EM Detections
                em_per_frame, em_per_minute, em_per_frame_labels, em_per_minute_labels = analyze_Bossu_predictions(em_detected, labels, offset, FPM, start_frame, thresholds = thresholds)
                em_per_frame_counter += np.asarray(em_per_frame["Type Errors"])
                em_per_minute_counter += np.asarray(em_per_minute["Type Errors"])
                if plots_per_file:
                    utils.make_metrics_plots(em_per_minute, thresholds, main_output_path, filename.replace(".mp4",".pdf"))
                
                ## Kalman Detections
                kalman_per_frame, kalman_per_minute, kalman_per_frame_labels, kalman_per_minute_labels = analyze_Bossu_predictions(kalman_detected, labels, offset, FPM, start_frame, thresholds = thresholds)
                kalman_per_frame_counter += np.asarray(kalman_per_frame["Type Errors"])
                kalman_sampled_counter += np.asarray(kalman_per_minute["Type Errors"])
                if plots_per_file:
                    utils.make_metrics_plots(kalman_per_minute, thresholds, main_output_path, filename.replace(".mp4","_kalman.pdf"))
                               
                
                row = []
                row.append(filename)
                row.append(total_frames)
                row.append(np.sum(em_detected))
                row.append(np.sum(kalman_detected))
                row.append(np.sum(em_detected)/total_frames * 100)
                row.append(np.sum(kalman_detected)/total_frames * 100)
                row.append(em_per_frame["Type Errors"][0][0])
                row.append(em_per_frame["Type Errors"][0][1])
                row.append(em_per_frame["Type Errors"][0][2])
                row.append(em_per_frame["Type Errors"][0][3])
                row.append(em_per_frame["Accuracy"][0])
                row.append(em_per_frame["F1-score"][0][0])
                row.append(em_per_frame["F1-score"][0][1])
                row.append(em_per_frame["MCC"][0])
                row.append(kalman_per_frame["Type Errors"][0][0])
                row.append(kalman_per_frame["Type Errors"][0][1])
                row.append(kalman_per_frame["Type Errors"][0][2])
                row.append(kalman_per_frame["Type Errors"][0][3])
                row.append(kalman_per_frame["Accuracy"][0])
                row.append(kalman_per_frame["F1-score"][0][0])
                row.append(kalman_per_frame["F1-score"][0][1])
                row.append(kalman_per_frame["MCC"][0])
                writer.writerow(row)

                assert (em_per_minute_labels == kalman_per_minute_labels).all(), "The minute labels for EM and Kalman are not the same!"
                assert (em_per_frame_labels == kalman_per_frame_labels).all(), "The frame labels for EM and Kalman are not the same!"

                label_total_per_minute += len(em_per_minute_labels)
                label_pos_per_minute += sum(em_per_minute_labels)
                label_total_per_frame += len(em_per_frame_labels)
                label_pos_per_frame += sum(em_per_frame_labels)
        
        # Calculate all metrics based on the accumlated type errors
        total_em_per_minute = calculate_classification_metrics_full_dataset(em_per_minute_counter, thresholds)
        total_em_per_frame = calculate_classification_metrics_full_dataset(em_per_frame_counter)
        total_kalman_per_minute  = calculate_classification_metrics_full_dataset(kalman_sampled_counter, thresholds)
        total_kalman_per_frame = calculate_classification_metrics_full_dataset(kalman_per_frame_counter)

        # Make plots of the different metrics
        utils.make_metrics_plots(total_em_per_minute, thresholds, main_output_path, "overall_em.pdf")
        utils.make_metrics_plots(total_kalman_per_minute, thresholds, main_output_path, "overall_kalman.pdf")
        
        # Save results
        with open(os.path.join(main_output_path,'evaluation_information.txt'), 'w') as f: 
            f.write("Metrics (EM) per frame: {}\n".format(total_em_per_frame))
            f.write("Metrics (Kalman) per frame: {}\n\n".format(total_kalman_per_frame))
            f.write("{} % Rain labels (Per minute)\n".format(label_pos_per_minute/label_total_per_minute * 100))
            f.write("{} rainy out of {} (Per minute)\n\n".format(label_pos_per_minute, label_total_per_minute))
            f.write("{} % Rain labels (Per frame)\n".format(label_pos_per_frame/label_total_per_frame * 100))
            f.write("{} rainy out of {} (Per frame)\n\n".format(label_pos_per_frame, label_total_per_frame))
            f.write("Label file used: {}\n".format(label_file))           
            f.write("Method used: Bossu\n")
            f.write("Label type used: {}".format(label_type))
        
        
        row = []
        row.append("Total")
        row.append("")
        row.append("")
        row.append("")
        row.append("")
        row.append("")
        row.append(total_em_per_frame["Type Errors"][0][0])
        row.append(total_em_per_frame["Type Errors"][0][1])
        row.append(total_em_per_frame["Type Errors"][0][2])
        row.append(total_em_per_frame["Type Errors"][0][3])
        row.append(total_em_per_frame["Accuracy"][0])
        row.append(total_em_per_frame["F1-score"][0][0])
        row.append(total_em_per_frame["F1-score"][0][1])
        row.append(total_em_per_frame["MCC"][0])
        row.append(total_kalman_per_frame["Type Errors"][0][0])
        row.append(total_kalman_per_frame["Type Errors"][0][1])
        row.append(total_kalman_per_frame["Type Errors"][0][2])
        row.append(total_kalman_per_frame["Type Errors"][0][3])
        row.append(total_kalman_per_frame["Accuracy"][0])
        row.append(total_kalman_per_frame["F1-score"][0][0])
        row.append(total_kalman_per_frame["F1-score"][0][1])
        row.append(total_kalman_per_frame["MCC"][0])
        writer.writerow(row)




if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes a set of output files from the Bossu exe file and compares them to the provided labels file. Calculates a plethora of binary classification meassures")
    ap.add_argument("-inputFolder", "--inputFolder", type=str, required = True,
                    help="Path to the main folder holding all the csv files to be evaluated")
    ap.add_argument("-labelFile", "--labelFile", type=str, required = True,
                    help="Path to the label file to be used")
    ap.add_argument("-outputFolder", "--outputFolder", type=str, required = True,
                    help="Path to the folder where the output plots and csv file should be saved")
    ap.add_argument('--filePlots', action="store_true",
                    help='saves metrics plots per input file')
    
    args = vars(ap.parse_args())
    
    evaluateBossu(args)