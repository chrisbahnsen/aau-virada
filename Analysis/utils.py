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

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt


def load_labels(filename):
    """
    Loads a JSON file containing the labels
    JSON data is saved into a global dict, and the function returns the corresponding function to get a specific label
    
    Input:
        filename: name of the JSON file
    
    Output:
        label function
        label_dict: Dict containig the label dicts for each video
    """

    with open(filename) as data_file:
        label_dict = json.load(data_file)

    keys = list(label_dict.keys())
    tmp = label_dict[keys[0]]["labels"]

    return label_dict
    

def read_yaml(file):
    """
    Takes a YAML formatted file as input, and returns it as dictionary
    
    Input:
        file : File path to the input file (Assumed to have '%YAML:1.0' as its first line)
        
    Output:
        Returns a python dictionary containing the elements in the YAML input file
    """
    with open(file, 'r') as fi:
        fi.readline()   #Skips the %YAML:1.0 on the first line
        return yaml.load(fi)



def plot_graph(x, y, x_label, y_label, title, x_range = (-0.05, 1.05), y_range = (-0.05, 1.05), legend = None, output_dir="", output_filename=""):
    ''' 
    Makes a line plot

    Input:
        x: x-values
        y: y-values. This can be a numpy array. (assumed to all use the same x axis values)
        x_label: Label of the x axis
        y_lbael: Label of the y axis
        title: Plot title
        x_range: Range of the x axis
        y_range: Range of the y axis
        legend: Whether to include a legend or not
        output_dir: Directory where the plot should be saved
        output_filename: Filename of the plot
    '''
    
    plt.clf()
    plt.figure()    
    if type(y) is np.ndarray:
        for i in range(y.shape[0]):
            plt.plot(x,y[i])
    else:
        plt.plot(x, y)
    plt.grid()
    plt.title(title)
    plt.ylim(*y_range)
    plt.xlim(*x_range)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:   
        plt.legend(legend,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')
    plt.close()
    

def make_metrics_plots(metric_dict, thresholds, output_folder, output_filename):
    '''
    Takes a dictionary containing the different binary-classification metrics, and creates a set of plot for each of them

    Input:
        metric_dict: Contains the different binary classification metrics
        thresholds: Threshold values used to calculate the different metric values
        output_folder: Folder where the output should be saved
        output_filename: Base filename. Assumed to end on .pdf
    '''
    
    TPR = [x[0] for x in metric_dict["Type Rates"]]
    TNR = [x[1] for x in metric_dict["Type Rates"]]
    FPR = [x[2] for x in metric_dict["Type Rates"]]
    FNR = [x[3] for x in metric_dict["Type Rates"]]
    NPV = [x[1] for x in metric_dict["Predictive Values"]]
    Precision = [x[0] for x in metric_dict["Precision-Recall"]]
    Recall = [x[1] for x in metric_dict["Precision-Recall"]]
    f_score = [x[0] for x in metric_dict["F1-score"]]
    inv_f_score = [x[1] for x in metric_dict["F1-score"]]
    acc = metric_dict["Accuracy"]
    inf = metric_dict["Informedness"]
    marked = metric_dict["Markedness"]
    MCC = metric_dict["MCC"]
    
    plot_graph(FPR, TPR, "FPR", "TPR", "Receiver Operating Charateristic", output_dir = output_folder, output_filename = output_filename.replace(".pdf","_ROC.pdf"))
    
    plot_graph(Recall, Precision, "Recall", "Precision", "Precision-Recall (TP)", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_PR.pdf"))
    
    plot_graph(TNR, NPV, "Inv. Recall", "Inv. Precision", "Precision-Recall (TN)", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_PRInv.pdf"))
    
    plot_graph(thresholds, f_score, "Threshold", "F1-Score (TP)", "F1-Score (TP)", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_F1S.pdf"))

    plot_graph(thresholds, inv_f_score, "Threshold", "F1-Score (TN)", "F1-Score (TN)", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_F1SInv.pdf"))

    plot_graph(thresholds, acc, "Threshold", "Accuracy", "Accuracy", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_Acc.pdf"))

    plot_graph(thresholds, inf, "Threshold", "Informedness", "Informedness", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_Inf.pdf"))
    
    plot_graph(thresholds, marked, "Threshold", "Markedness", "Markedness", output_dir = output_folder, output_filename =output_filename.replace(".pdf","_Mark.pdf"))
    
    plot_graph(thresholds, MCC, "Threshold", "MCC", "Matthews Correlation Coefficient", (-0.05, 1.05), (-1.05, 1.05), output_dir = output_folder, output_filename = output_filename.replace(".pdf","_MCC.pdf"))
    
    plot_graph(thresholds, np.asarray([TPR, TNR]), "Threshold", "True Rate", "Type Rates", legend = ("TPR", "TNR"), output_dir = output_folder, output_filename =output_filename.replace(".pdf","_TR.pdf"))
    
    plot_graph(thresholds, np.asarray([FPR, FNR]), "Threshold", "False Rate", "Type Rates", legend = ("FPR", "FNR"), output_dir = output_folder, output_filename =output_filename.replace(".pdf","_FR.pdf"))
   
    
def get_frame_label(labels, offset, FPM, frame_num):
    """
    Retrives the correct label for a frame, depending on the frames per minute and the offset of the video

    Input:
        labels: List containing all the labels per minute for the video
        offset: How many frames left of the starting minute e.g. 16:00:45, has 15 seconds left
                This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
                These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:01:00
        FPM: The amount of frames in a minute in the video
        frame_num: frame number of the first frame in the sequence

    Output:
        per-minute rain label
        label index - corresponds to the minute
    """

    # Logic flow to determine which label it should use (which minute)
    if frame_num <= offset and offset > 0:
        # If there is an offset (i.e. the video starts in the middle of a minute) and the frame number is below or equal this offset
        # return: Label of the first minute in the video
        ind = 0
    else:
        if offset > 0:
            # If there is an offset
            # Subtract offset from frame number, and divide by FPM, and then round up to get our index. e.g.
            # offset = 300, frame_num = 400, FPM = 1800, ind = 400-300 / 1800 = 0.055 -> 1
            ind = int(np.ceil((frame_num-offset)/FPM))
        else:
            # If there is not an offset
            # Take frame number, and divide by FPM, and then round down to get our index. e.g.
            # frame_num = 400, FPM = 1800,   ind = 400 / 1800 = 0.2222 -> 0
            ind = int(np.floor(frame_num/FPM))

    ind = min(ind, len(labels)-1)
    return labels[ind], ind
