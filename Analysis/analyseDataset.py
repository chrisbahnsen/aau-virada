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
import utils
import csv
import argparse
import numpy as np


def analyseDataset(args):
    '''
    Calculates the statistics of the videos in the provided csv file. This consists of the amount of video file, amount of frames, minutes and the percentage of rain frames.

    Input:
        args:
            - labelFile: Path JSON file containing the per minute labels
            - videoList: Path to CSV file containing which videos are a part of the dataset
            - outputpath: Path for the the resulting text file
    '''

    # Load parameters
    label_file = args["labelFile"]
    vid_list = args["videoList"]
    output_path = args["outputPath"]
    root_output_name = vid_list[:-4]

    # Determine if it is mechanical or laser labels. TODO : make this an input variable
    label_type = "mechanical"
    if "laser" in label_file:
        label_type = "laser"

    # Load label dict
    label_dict = utils.load_labels(label_file)

    label_total = 0
    label_pos = 0
    ps_label_total = 0
    ps_label_pos = 0
    vidCount = 0
    totalFrames = 0

    # Go through each video in the provided csv file
    with open(vid_list, 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvReader:
            if len(row) == 0:
                continue

            # Only investigate mp4 files
            filename = row[0].strip()
            if os.path.splitext(filename)[-1] != ".mp4":
                continue
            
            # Get per-minute label information
            dict_ind = label_dict[os.path.basename(filename)]
            offset = dict_ind["frameOffset"]    # How many frames left of the starting minute e.g. 16:00:45, has 15 seconds left 
                                                # This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
                                                # These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:00:01
            FPM = dict_ind["FPM"]  # Frames per minute
            labels = dict_ind["labels"] # List of labels per minute
            frameCount = dict_ind["frameCount"]      
            
            # Analyze data
            per_sample_labels = []

            # Get per frame labels
            for i in range(frameCount):
                label, minute = utils.get_frame_label(labels, offset, FPM, i)
                per_sample_labels.append(label)
            per_sample_labels = np.asarray(per_sample_labels).astype(np.bool)

            labels = np.asarray(labels).astype(np.bool)
            
            label_total += len(labels)  # Amount of per-minute labels
            label_pos += sum(labels)    # Amount of rain per-minute labels
            ps_label_total += len(per_sample_labels) # Amount of per-frame labels
            ps_label_pos += sum(per_sample_labels) # Amount of per-frame rain labels
            totalFrames += frameCount # Amount of frames
            vidCount += 1 # Amount of videos
                
        
        with open(root_output_name + "_" +label_type + "_dataset.txt", 'w') as f: 
            f.write("{} % Rain labels (Per minute)\n".format(label_pos/label_total * 100))
            f.write("{} rainy out of {} (Per minute)\n".format(label_pos, label_total))
            f.write("{} % Rain labels (Per frame)\n".format(ps_label_pos/ps_label_total * 100))
            f.write("{} rainy out of {} (Per frame)\n".format(ps_label_pos, ps_label_total))
            f.write("Amount of videos: {}\n".format(vidCount))
            f.write("Amount of frames: {}\n".format(totalFrames))
            f.write("Labels file: {}\n".format(args["labelFile"]))
            f.write("Video csv file: {}\n".format(args["videoList"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Computes statistics for the supplied dataset")
    ap.add_argument("-videoList", "--videoList", type=str, required = True,
                    help="Path to the csv file containig all video names to be included")
    ap.add_argument("-labelFile", "--labelFile", type=str, required = True,
                    help="Path to the label file to be used")
    ap.add_argument("-outputPath", "--outputPath", type=str, required = True,
                    help="Path to the output directory")
    
    args = vars(ap.parse_args())
    
    analyseDataset(args)