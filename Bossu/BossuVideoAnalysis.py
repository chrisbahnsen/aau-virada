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
import subprocess
import csv
import argparse
from BossuCSVAnalysis import analyseBossuCSVData

def videoAnalysis(args):
    """
    Takes a csv file as input where the path to different videos which has to be analyzed with the BossuRainGauge C++ project is defined.
    
    Input:
        args:
            - videoPathFile: Path to the csv file containing all the video file paths and metadata of the video (Filepath; Start time (DD-MM-YYYY HH:MM:SS); Latitude; Longitude; FPS)
            - exeFilePath: Path to the c++ exe file
            - rainGaugeFilePath: Path to the rain gauge data file
            - outputPath: Path to the main output folder
            - debugFlag: Whether to set debug flag in the c++ exe file
            - verboseFlag: Whether to set verbose flag in the c++ exe file
            - saveImageFlag: Whether to set saveImage flag in the c++ exe file
            - saveSettingsFlag: Whether to set saveSettings flag in the c++ exe file
            - startDateTime: Start time of the video (DD-MM-YYY HH:MM:SS)
            - c: List of integer values for the "c" parameter to test
            - dm: List of float values for the "dm" parameter to test
            - em: List of integer values for the "emMaxIterations" parameter to test
            - gof: List of float values for the "maxGoFDifference" parameter to test. Should be between 0 and 1
            - surface: List of float values for the "minimumGaussianSurface" parameter to test. Should be between 0 and 1
            - maxBlobSize: List of integer values for the "maximumBlobSize" parameter to test
            - minBlobSize: List of integer values for the "minimumBlobSize" parameter to test
    """    
    

    for key in ["c", "em", "surface", "gof", "dm", "maxBlobSize", "minBlobSize"]:
        if type(args[key]) != list:
            args[key] = [args[key]]

    # Set up the arguments passed to the cpp executable, based on input args
    exe_args_base = []
    exe_args_base.append(args["exeFilePath"])
    exe_args_base.append("--d="+str(args["debugFlag"]))
    exe_args_base.append("--v="+str(args["verboseFlag"]))
    exe_args_base.append("--i="+str(args["saveImageFlag"]))
    exe_args_base.append("--s="+str(args["saveSettingsFlag"]))
    
    
    # Setup output directory
    OUTPUT_DIR_BASE = os.path.abspath(os.path.dirname(__file__))
    if args["outputPath"] != "":
        OUTPUT_DIR_BASE = args["outputPath"]


    # Go through each video filepath, run the cpp executable, and analyze output data, and compare to rain gauge info
    with open(args["videoPathFile"]) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        
        for row in reader:
            video = row[0]
            video = video.replace("\\","/")
            fileName = video.split("/")[-1]
            filePath = video[:-len(fileName)]
            
            
            OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, fileName[:8])
            
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            if not os.path.exists(video):
                print("{} does not exist".format(video))
            
            for em in args["em"]:
                for c in args["c"]:
                    for dm in args["dm"]:
                        for maxBlobSize in args["maxBlobSize"]:     
                            for minBlobSize in args["minBlobSize"]:
                                for gof in args["gof"]:
                                    for surface in args["surface"]:     
                
                                        FILE_DIR = os.path.join(OUTPUT_DIR, "c_{}_em_{}_dm_{}_maxbs_{}_minbs_{}_gof_{}_surface_{}".format(c, em, dm, maxBlobSize, minBlobSize, gof, surface))
                                        print("Output dir: {}".format(FILE_DIR))
                                        if not os.path.exists(FILE_DIR):
                                            os.makedirs(FILE_DIR)
                                
                                
                                        settingsName = "Settings_c_{}_em_{}_dm_{}_maxbs_{}_minbs_{}_gof_{}_surface_{}.txt".format(c, em, dm, maxBlobSize, minBlobSize, gof, surface)
                                        settingsFile = os.path.join(FILE_DIR,settingsName)
                                        
                                        if not os.path.exists(settingsFile):
                                            with open(settingsFile, "w") as file:
                                                file.write("%YAML:1.0\n")
                                                file.write("c: {}\n".format(c))
                                                file.write("minimumBlobSize: {}\n".format(minBlobSize))
                                                file.write("maximumBlobSize: {}\n".format(maxBlobSize))
                                                file.write("dm: {}\n".format(dm))
                                                file.write("maxGoFDifference: {}\n".format(gof))
                                                file.write("minimumGaussianSurface: {}\n".format(surface))
                                                file.write("emMaxIterations: {}}\n".format(em))
                                                file.write("saveImg: {}\n".format(args["saveImageFlag"]))
                                                file.write("verbose: {}\n".format(args["verboseFlag"]))
                                                file.write("debug: {}\n".format(args["debugFlag"]))
                                
                                        # Run cpp executable
                                        exe_args = exe_args_base.copy()
                                        exe_args.append("--of="+FILE_DIR)
                                        exe_args.append("--fileName=" + fileName)
                                        exe_args.append("--filePath=" + filePath)
                                        exe_args.append("--settingsFile=" + settingsName)
                                        subprocess.call(exe_args, creationflags=subprocess.CREATE_NEW_CONSOLE)
                                        
                                        # Run analysis on output data
                                        CSV_analysis = {}
                                        CSV_analysis["dataFilePath"] = FILE_DIR+"/"+fileName+"_Results.csv"
                                        CSV_analysis["outputPath"] =  FILE_DIR
                                        CSV_analysis["gaugeFilePath"] = args["rainGaugeFilePath"]
                                        CSV_analysis["startDateTime"] = row[1]
                                        CSV_analysis["latitude"] = float(row[2])
                                        CSV_analysis["longitude"] = float(row[3])
                                        CSV_analysis["frameRate"] = int(row[4])
                                        analyseBossuCSVData(CSV_analysis)
                        
        


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes a .csv file where each row contains the filepath to a video, as well as parameters related to the video, and then runs the BossuRainGauge C++ project on it. The output is then evaluated and compared against rain measurements from a nearby rain gauge")
    ap.add_argument("-videoPathFile", "--videoPathFile", type=str, required = True,
                    help="Path to the .csv data file, which contains all filepaths to the videos, and additional information. The filepaths should be the complete path to the file, including the filename itself. The .csv file should be formatted as follows (without a header!):\nFilepath;Start time (DD-MM-YYYY HH:MM:SS);Latitude;Longitude;FPS")
    ap.add_argument("-exeFilePath", "--exeFilePath", type=str, required = True,
                    help="Path to the exe file of the BossuRainGauge C++ project. Provide the entire filepath including the filename")
    ap.add_argument("-rainGaugeFilePath", "--rainGaugeFilePath", type=str, required = True,
                    help="Path to the rain gauge overview file. Provide the entire filepath including the filename")
    ap.add_argument("-outputPath", "--outputPath", type=str, required = True,
                    help="Path to main output folder. If provided a folder will be made containing the output .csv and settings file. Else it will be saved in a folder in where the script is placed. The path given should be the entire filepath")
    
    ap.add_argument("-debugFlag", "--debugFlag", type=int, default = 0,
                    help="States whether the debug flag should be set or not. Not set if equal to 0")
    ap.add_argument("-verboseFlag", "--verboseFlag", type=int, default = 0,
                    help="States whether the verbose flag should be set or not. Not set if equal to 0")
    ap.add_argument("-saveImageFlag", "--saveImageFlag", type=int, default = 0,
                    help="States whether the saveImage flag should be set or not. Not set if equal to 0")
    ap.add_argument("-saveSettingsFlag", "--saveSettingsFlag", type=int, default = 0,
                    help="States whether the saveSettings flag should be set or not. Not set if equal to 0")

    ap.add_argument("-c", "--c", nargs = "+", type=int, default = 3,
                    help="Sets the 'c' parameter in the Bossu algorithm. Takes a list a list as input")
    ap.add_argument("-minBlobSize", "--minBlobSize", nargs = "+", type=int, default = 4,
                    help="Sets the 'minimumBlobSize' parameter in the Bossu algorithm. Takes a list as input")
    ap.add_argument("-maxBlobSize", "--maxBlobSize", nargs = "+", type=int, default = 200,
                    help="Sets the 'maximumBlobSize' parameter in the Bossu algorithm. Takes a list as input")
    ap.add_argument("-dm", "--dm", nargs = "+", type=float, default = 0.5,
                    help="Sets the 'dm' parameter in the Bossu algorithm. Takes a list as input")
    ap.add_argument("-gof", "--gof", nargs = "+", type=float, default = 0.19,
                    help="Sets the 'maxGoFDifference' parameter in the Bossu algorithm. Should be a value between 0 and 1. Takes a list as input")
    ap.add_argument("-surface", "--surface", nargs = "+", type=float, default = 0.4,
                    help="Sets the 'minimumGaussianSurface' parameter in the Bossu algorithm. Should be a value between 0 and 1. Takes a list as input")
    ap.add_argument("-em", "--em", nargs = "+", type=int, default = 100,
                    help="Sets the 'emMaxIterations' parameter in the Bossu algorithm. Should be a value between 0 and 1. Takes a list as input")
    


    args = vars(ap.parse_args())
    
    videoAnalysis(args)