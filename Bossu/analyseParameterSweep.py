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
import argparse

sys.path.append('../')
from Analysis.utils import read_yaml


def analyseBossuParameterSweep(args):
    '''
    Goes through the input dir and analyses and collects the amount of predicted rain per parameter configuration for the Bossu rain detector
    It is assumed the input dir is structured with a folder per analyzed video, with a separate folder for each parameter combination inside each of these folders
    The output csv file is named after the analyzed video.

    Input:
        inputFolder: Path to the input folder
        outputFolder: Path to the folder where the output will be saved.
        hosCheck: Boolean indicated whether the goodness-of-fit and surface value checks should be performed
    '''
    
    main_path = args["inputFolder"]
    main_output_path = args["outputFolder"]
    hosCheck = args["hosCheck"]

    if not os.path.exists(main_output_path):
        os.makedirs(main_output_path)
            
    # Go through each child folder in the inpur folder
    for dirs in os.listdir(main_path):
        
        dir_path = os.path.join(main_path, dirs)
        print(dirs)
        
        # Initialize output csv file
        output_path = os.path.join(main_output_path, dirs + "_collected.csv")    
        with open(output_path, 'w', newline = "") as csvWriteFile:
            writer = csv.writer(csvWriteFile, delimiter=";")
            
            # Write the headers in the new csv file
            firstrow = []
            firstrow.append("c")
            firstrow.append("dm")
            firstrow.append("Max BLOB size")
            firstrow.append("GoF Thresh")
            firstrow.append("Surface Thresh")
            firstrow.append("Total Frames")
            firstrow.append("EM Rain Frames")
            firstrow.append("Kalman Rain Frames")
            firstrow.append("EM %")
            firstrow.append("Kalman %")
            
            writer.writerow(firstrow)
            
            # Go through each child-child folder, each containig results of a parameter combination
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                files =  os.listdir(subdir_path)
                
                csv_file = ""
                settings_file = ""
                    
                # Read result csv and settings txt file
                for file in files:
                    if os.path.splitext(file)[-1] == ".csv":
                        csv_file = os.path.join(subdir_path, file)
                    if os.path.splitext(file)[-1] == ".txt":
                        settings_file = os.path.join(subdir_path, file)
                
                print(subdir)
                print(files)
                print(csv_file)
                print()
                
                ###### LOAD BOSSU OUTPUT ######
                #Load the supplied csv file
                rain_dataframe = pd.read_csv(csv_file, sep=";")
                
                # Load the differnet parts of the csv file
                rainDetections = rain_dataframe[rain_dataframe.columns[12:14]]
                rainValues = rainDetections.values
                
                
                #Read and set parameters in csv file
                settings = read_yaml(settings_file)
                c = settings["c"]
                maxBlobSize = settings["maximumBlobSize"]
                dm = settings["dm"]
                gof = settings["maxGoFDifference"]
                surface = settings["minimumGaussianSurface"]
                
                if hosCheck:
                    gof_passed = rain_dataframe["Goodness-Of-Fit Value"].values <= gof
                    kalman_surface = rain_dataframe["kalmanGaussMixProp"].values >= surface
                    em_surface = rain_dataframe["GaussMixProp"].values >= surface
                    
                    em_passed = np.sum(np.logical_and(gof_passed, em_surface))
                    kalman_passed = np.sum(np.logical_and(gof_passed, kalman_surface))
                else:
                    em_passed = np.sum(rainValues[:,0])
                    kalman_passed = np.sum(rainValues[:,1])
                
                
                row = []
                row.append(c)
                row.append(dm)
                row.append(maxBlobSize)
                row.append(gof)
                row.append(surface) 
                row.append(rainValues.shape[0])
                row.append(em_passed) # Amount of detected rain frames using EM 
                row.append(kalman_passed) # Amount of detected rain frames using Kalman
                row.append(em_passed/rainValues.shape[0] * 100) # Ratio of detected rain frames using EM
                row.append(kalman_passed/rainValues.shape[0] * 100) # Ratio of detected rain frames using Kalman
                
                writer.writerow(row)
                
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Analyses Bossu parameter sweep")
    ap.add_argument("-inputFolder", "--inputFolder", type=str, required = True, default=".\ParameterSweep",
                    help="Path to the csv file containig all video names to be included")
    ap.add_argument("-outputFolder", "--outputFolder", type=str, required = True, default=".\ParameterSweepOutput",
                    help="Path to the output directory")
    ap.add_argument("-hosCheck", "--hosCheck", action="store_true",
                    help="Whether to check whether the GoF and Surface values passes the thresholds. used for the GoF and Surface parameter sweep")
    
    args = vars(ap.parse_args())
    
    analyseBossuParameterSweep(args)