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
import argparse
import os
import csv
import cv2
import time

sys.path.append('../')
from Analysis.utils import read_yaml

    

def gofAndSurfaceSweepFile(dataframe, settings_path, output_path, fileName, gof_thresh = 0.06, surface_thresh = 0.35):
    """
    Determine the Kalman predicted values for all the detections in the input file, using the provided threshold values for the Goodness-of-Fit and HOS Surface thresholds
    Saves a new csv file with the new Kalman predictions

    Input:
        dataframe: Pandas dataframe containing the csv data of the to be evaluated video
        settings_path: Path to the original settings file
        output_path: Where to save the output csv file with the updated Kalman predictions
        fileName: Filename of the investigated file
        gof_thresh: Threshold for the Goodness of Fit test
        surface_thresh: Threshold of the HOS Surface
    """
    
    old_settings = read_yaml(settings_path)
    c = old_settings["c"]
    maxBlobSize = old_settings["maximumBlobSize"]
    dm = old_settings["dm"]
    verbose = old_settings["verbose"]
    saveImage = old_settings["saveImg"]
    debugFlag = old_settings["debug"]
    
    settingsName = "Settings_{}_c_{}_dm_{}_mbs_{}_gof_{:.2f}_surf_{:.2f}.txt".format(fileName,c,dm,maxBlobSize, gof_thresh, surface_thresh)
    settingsFile = os.path.join(output_path, settingsName)
    
    ## Write settings file with parameters
    with open(settingsFile, "w") as file:
        file.write("%YAML:1.0\n")
        file.write("c: {}\n".format(c))
        file.write("minimumBlobSize: 4\n")
        file.write("maximumBlobSize: {}\n".format(maxBlobSize))
        file.write("dm: {}\n".format(dm))
        file.write("maxGoFDifference: {:.4f}\n".format(gof_thresh))
        file.write("minimumGaussianSurface: {:.4f}\n".format(surface_thresh))
        file.write("emMaxIterations: 100\n")
        file.write("saveImg: {}\n".format(saveImage))
        file.write("verbose: {}\n".format(verbose))
        file.write("debug: {}\n".format(debugFlag))
        
    
    # Initialize kalman filter for the HOS distribution parameters, using a constant velocity model
    kalman_mean = 0.0
    kalman_stddev = 0.0
    kalman_mixprop = 0.0
    
    KF = cv2.KalmanFilter(6,3,0, cv2.CV_64F)

    KF.transitionMatrix = np.array([[1., 0., 0., 1., 0., 0.], 
                                    [0., 1., 0., 0., 1., 0.],
                                    [0., 0., 1., 0., 0., 1.],
                                    [0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 1.]])
    KF.measurementMatrix = np.array([[1., 0., 0., 0., 0., 0.], 
                                    [0., 1., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0.],])
    KF.processNoiseCov = np.eye(6) * 0.01
    KF.measurementNoiseCov = np.eye(3) * 0.1
    KF.errorCovPost = np.eye(6)
    
    

    file_path = os.path.join(output_path, fileName)
        
    with open("{}_c_{}_dm_{}_mbs_{}_gof_{:.2f}_surf_{:.2f}.csv".format(file_path, c, dm, maxBlobSize, gof_thresh, surface_thresh), 'w', newline = "") as csvWriteFile:
        writer = csv.writer(csvWriteFile, delimiter=";")
        
        # Write the headers in the new csv file
        firstrow = []
        firstrow.append("settingsFile")
        firstrow.append("InputVideo")
        firstrow.append("Frame")
        firstrow.append("GaussMean")
        firstrow.append("GaussStdDev")
        firstrow.append("GaussMixProp")
        firstrow.append("Goodness-Of-Fit Value")
        firstrow.append("kalmanGaussMean")
        firstrow.append("kalmanGaussStdDev")
        firstrow.append("kalmanGaussMixProp")
        firstrow.append("Rain Intensity")
        firstrow.append("Kalman Rain Intensity")
        firstrow.append("EM Rain Detected")
        firstrow.append("Kalman Rain Detected")
         
        writer.writerow(firstrow)
        
        ## Go throuh each detection and evaluate according to the set threshold parameters
        for index, row in dataframe.iterrows():
            em_mean = row[" GaussMean"]
            em_stddev = row["GaussStdDev"]
            em_mixprop = row["GaussMixProp"]
            em_rainintensity_csv = row["Rain Intensity"]
            em_rainDetected = False
            kalman_rainDetected = False
            kalman_rainIntensity = 0.0
            
            ## Sanity check of whether the EM estimated mean is above 0
            if(em_mean >= 0):
                gof_val = row["Goodness-Of-Fit Value"]
                
                kalman_mixprop_csv = row["kalmanGaussMixProp"]
                kalman_rainintensity_csv = row["Kalman Rain Intensity"]
                
                # Estimate the surface value from the determined rain intensity, since the surface value is not directly saved.
                em_div = em_mixprop > 0.0
                kalman_div = kalman_mixprop_csv > 0.0
                surface = 0
                
                # If rain is detected for both EM and kalmna, take the average of the estimated surface values
                # Else take the value of the single estimate available
                if em_div and kalman_div:
                    em_surface = em_rainintensity_csv/em_mixprop
                    kalman_surface = kalman_rainintensity_csv/kalman_mixprop_csv
                    
                    surface = (kalman_surface + em_surface)/2
                elif em_div:
                    surface = em_rainintensity_csv/em_mixprop
                elif kalman_div:
                    surface = kalman_rainintensity_csv/kalman_mixprop_csv
                else:
                    surface = 0.
                
                # Check whether the Goodness-Of-Fitness test is passed
                gof_passed = gof_val <= gof_thresh
                
                if gof_passed:
                    kalmanPredict = KF.predict()
                    measurement = np.array([em_mean, em_stddev, em_mixprop])
                    measurement = measurement.reshape((-1,1))
                    estimated = KF.correct(measurement)
                    
                    kalman_mean = estimated[0]
                    kalman_stddev = estimated[1]
                    kalman_mixprop = estimated[2]
                    
                # Check whether the Surface test is passed
                em_surface_passed = em_mixprop >= surface_thresh
                kalman_surface_passed = kalman_mixprop >= surface_thresh
                
                # Determine whether rain is detected or not
                em_rainDetected = gof_passed and em_surface_passed
                kalman_rainDetected = gof_passed and kalman_surface_passed
                kalman_rainIntensity = surface * kalman_mixprop      
            
            csv_row = []
            csv_row.append(settingsFile)
            csv_row.append(row["InputVideo"])
            csv_row.append(row[" Frame#"])
            csv_row.append(em_mean)
            csv_row.append(em_stddev)
            csv_row.append(em_mixprop)
            csv_row.append(row["Goodness-Of-Fit Value"])
            csv_row.append(float(kalman_mean))
            csv_row.append(float(kalman_stddev))
            csv_row.append(float(kalman_mixprop))
            csv_row.append(em_rainintensity_csv)
            csv_row.append(float(kalman_rainIntensity))
            csv_row.append(int(em_rainDetected))
            csv_row.append(int(kalman_rainDetected))
            writer.writerow(csv_row)
        
    
def gofAndSurfaceSweep(args):
    """
    Main loop for analysing several files with new Goodness-Of-Fit and HOS Surface thresholds.
    Assumes the input is structured per date, and that per date there is a folder per instance (e.g. per hour), each containing the produced csv and settings files.

    Input:
        args:
            - inputFolder: Path to the input folder
            - outputFolder: Path to the output folder
            - minGOF: Minimum GOF threshold in the range [0; 100]
            - maxGOF: Maximum GOF threshold in the range [0; 100] (Not included)
            - GOFsteps: The step size used between minGOF and maxGOF. Should be integer
            - minSurface: Minimum surface threshold value in the range [0; 100]
            - maxSurface: Maximum surface threshold value in the range [0; 100] (Not included)
            - SurfaceSteps: The step size used between minSurface and maxSurface. Should be integer
    """

    main_path = args["inputFolder"]
    main_output_path = args["outputFolder"]

    if not os.path.exists(main_output_path):
        os.makedirs(main_output_path)
    
    for dirs in os.listdir(main_path):
        dir_path = os.path.join(main_path, dirs)
        print(dirs)
        
        output_path = os.path.join(main_output_path, dirs)    

        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            files =  os.listdir(subdir_path)
            
            csv_file = ""
            settings_file = ""
                
            # Get csv and settigns file paths
            for file in files:
                if os.path.splitext(file)[-1] == ".csv":
                    csv_file = os.path.join(subdir_path, file)
                if os.path.splitext(file)[-1] == ".txt":
                    settings_file = os.path.join(subdir_path, file)
            
            print(subdir)
            print(files)
            print(csv_file)
            print()
            
            # Create output folder
            subdir_output_path = os.path.join(output_path, subdir)
            if not os.path.exists(subdir_output_path):
                os.makedirs(subdir_output_path)
            
            
            ###### LOAD BOSSU OUTPUT ######
            #Load the supplied csv file
            rain_dataframe = pd.read_csv(csv_file, sep=";")

            for gof in range(args["minGOF"], args["maxGOF"], args["GOFsteps"]):
                gof /= 100
                for surface in range(args["minSurface"], args["maxSurface"], args["SurfaceSteps"]):
                    surface /= 100
                
                    _start = time.time()
                    gofAndSurfaceSweepFile(rain_dataframe, settings_file, subdir_output_path, dirs, gof, surface)
                    _end = time.time()
                    
                    print("GoF: {:.2f}, Surface Thresh: {:.2f}, Time: {:.2f}".format(gof, surface, _end-_start))
            
        



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes a set of output files from the Bossu C++ code and sweeps over the parameter thresholds affecting the kalman filter")
    ap.add_argument("-inputFolder", "--inputFolder", type=str, required = True, default = ".\ParameterSweep",
                    help="Path to the main folder holding all the csv files to be evaluated")
    ap.add_argument("-outputFolder", "--outputFolder", type=str, required = True, default=".\ParameterSweepGoFSurface",
                    help="Path to the folder where the output plots and csv file should be saved")
    ap.add_argument("-minGOF", "--minGOF", type=int, default = 1, required = True,
                    help="Min surface threshold value. Should be integer and in the range [0; 100]")
    ap.add_argument("-maxGOF", "--maxGOF", type=int, default = 21, required = True,
                    help="Max GOF threshold value. Should be integer and in the range [0; 100]. The value is not included")
    ap.add_argument("-GOFsteps", "--GOFsteps", type=int, default = 1, required = True,
                    help="Step size for the GOF threshold parameters. Should be an integer value")
    ap.add_argument("-minSurface", "--minSurface", type=int, default = 20, required = True,
                    help="Min surface threshold value. Should be integer and in the range [0; 100]")
    ap.add_argument("-maxSurface", "--maxSurface", type=int, default = 51, required = True,
                    help="Max surface threshold value. Should be integer and in the range [0; 100]. The value is not included")
    ap.add_argument("-SurfaceSteps", "--SurfaceSteps", type=int, default = 2, required = True,
                    help="Step size for the surface threshold parameters. Should be an integer value")

    args = vars(ap.parse_args())
    
    gofAndSurfaceSweep(args)