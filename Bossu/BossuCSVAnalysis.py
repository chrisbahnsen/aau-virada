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

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import datetime
from sklearn.metrics import auc

sys.path.append('../')
import Analysis.RainGauges as RG
from Analysis.utils import read_yaml

def plot_distribution_params(x_axis, gaussParams, kalmanGaussParams, threshold = 0.35, outputPath = ""):
    """
    Plots the Mean, Standard Deviation and Mixture Proportion obtained from the EM algorithm and Kalman filtering, against the frame count.
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        gaussParams : Panda DataFrame containing the EM estiamted Gaussian parameters of the rain streaks, with dimensions [n_frames, 3]
        kalmanGaussParams : Panda DataFrame containing the Kalman filtered Gaussian parameters of the rain streaks, with dimensions [n_frames, 3]
        threshold : The threshold value of the Gaussian Mixture Proportion used to determine whether to update Kalman filter or not
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf with 3 plots, where the EM estiamted and Kalman filtered parameters are compared against each other
    """
    
    gaussValues = gaussParams.values
    kalmanValues = kalmanGaussParams.values
    x_axis_min = np.min(x_axis)
    x_axis_max = np.max(x_axis)
    
    titles = ["Mean", "Standard Deviation", "Mixture Proportion"]
    lgnd = ["EM", "Kalman"]
    
    
    plt.figure(1,(15,15)) #second argument is size of figure in integers
    plt.clf()
    
    plt.suptitle("Distribution parameters")
    
    plt.subplot(3,1,1)
    plt.title(titles[0])
    plt.plot(x_axis,gaussValues[:,0],'b')
    plt.plot(x_axis,kalmanValues[:,0],'orange')
    plt.legend(lgnd, loc='lower right')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    
    plt.subplot(3,1,2)
    plt.title(titles[1])
    plt.plot(x_axis,gaussValues[:,1],'b')
    plt.plot(x_axis,kalmanValues[:,1],'orange')
    plt.legend(lgnd, loc='lower right')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    
    
    plt.subplot(3,1,3)
    plt.title(titles[2])
    plt.plot(x_axis,gaussValues[:,2],'b')
    plt.plot(x_axis,kalmanValues[:,2],'orange')
    
    horiz_line_data = np.array([threshold for i in range(len(x_axis))])
    plt.plot(x_axis, horiz_line_data, 'g--') 
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    lgnd.append("Threshold: {:.2f}".format(threshold))    
    plt.legend(lgnd, loc='lower right')
    
    
    #plt.show()
    plt.savefig(outputPath + "Distribution_Parameters.pdf", bbox_inches="tight")


def plot_rain_detection(x_axis, rainDetections, outputPath = ""):
    """
    Plots the calculated rain detection value against the number of frames. Both the rain intensity based on the EM estimated and Kalman filtered Mixture Proportion are plotted
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        rainDetections : Panda DataFrame containing the rain detections, with dimensions [n_frames, 2]
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf where the EM estiamted and Kalman filtered rain intensity are compared against each other
    """
    
    rainValues = rainDetections.values
    rainNames = list(rainDetections)
    
    plt.figure(2,(15,15)) #second argument is size of figure in integers
    plt.clf()

    plt.subplot(3,1,1)
    plt.title(rainNames[0] + " " + str(np.sum(rainValues[:,0])) + " / " + str(len(rainValues[:,0])))
    plt.scatter(x_axis, rainValues[:,0], c = 'b')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(np.min(x_axis), np.max(x_axis))
    
    
    plt.subplot(3,1,2)
    plt.title(rainNames[1] + " " + str(np.sum(rainValues[:,1])) + " / " + str(len(rainValues[:,1])))
    plt.scatter(x_axis, rainValues[:,1], c = 'orange')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(np.min(x_axis), np.max(x_axis))
        
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(outputPath + "Rain_Detections.pdf", bbox_inches="tight")
    
    
def plot_raw_rain_intensity(x_axis, rainIntensities, outputPath = ""):
    """
    Plots the calculated rain intensity value against the number of frames. Both the rain intensity based on the EM estimated and Kalman filtered Mixture Proportion are plotted
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        rainIntensities : Panda DataFrame containing the rain intensities, with dimensions [n_frames, 2]
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf where the EM estiamted and Kalman filtered rain intensity are compared against each other
    """
    
    rainValues = rainIntensities.values
    rainNames = list(rainIntensities)
    
    plt.figure(3,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    for i in range(rainValues.shape[1]):
        plt.plot(x_axis, rainValues[:,i], label = rainNames[i])
    plt.grid(True)
    plt.xlabel("Frame")
    plt.xlim(np.min(x_axis), np.max(x_axis))
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(outputPath + "Raw_Rain_Intensity.pdf", bbox_inches="tight")


def plot_avg_rain_intensity(datetimeRI, datetimeGauge, avgEM, avgKalman, gaugeMeasurrements, labels, outputPath = "", useSubplots = False, numTicks = 10):
    """
    Plots the calculated rain intensity value against the number of frames. Both the rain intensity based on the EM estimated and Kalman filtered Mixture Proportion are plotted
    
    Input:
        datetimeRI : List of datetime values for the Bossu rain intensities
        datetimeGauge : List of datetime values for the rain gauge measurements
        avgEM: EM estimated rain intensity averaged per second
        avgKalman: Kalman estiamte rain intensity averaged per second
        gaugeMeasurrements: Rain intensity from the rain gauge
        labels: List of the label names for plotting
        outputPath : File path for where the output pdf should be saved
        useSubplots: Whether to plot the rain gauge measurements in a separate subplot or in the same plot as the Bossu rain intensity
        numTicks: The number of ticks along the x-axis
        
    Output:
        Outputs a single pdf where the per second averaged EM estiamted and Kalman filtered rain intensity are compared against each other and rain data from the nearest rain gauge data
    """
    
    startTime = None
    endTime = None
    
    if datetimeRI[0] < datetimeGauge[0]:
        startTime = datetimeRI[0]
    else:
        startTime = datetimeGauge[0]
        
    if datetimeRI[-1] > datetimeGauge[-1]:
        endTime = datetimeRI[-1]
    else:
        endTime = datetimeGauge[-1]
    
    
    timeDiff = endTime - startTime
    totalSeconds = int(timeDiff.total_seconds())
    x_axis = range(totalSeconds+1)
    
    timeStamps = [startTime + datetime.timedelta(seconds=i) for i in x_axis]
    
    x_axisRI = [i for i, date in enumerate(timeStamps) if date in datetimeRI]
    
    x_axisGauge = [i for i, date in enumerate(timeStamps) if date in datetimeGauge]
    
    x_axis_timeStamps = [i.strftime("%H:%M:%S") for i in timeStamps]   
    
        
    def format_func(values, tick_number):
        if int(values) in x_axis:
            return x_axis_timeStamps[int(values)]
        else:
            return ""
        
    
    plt.figure(4,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    if useSubplots:
        
        plt.plot(x_axisRI, avgEM, label = labels[0], color = "b")
        plt.plot(x_axisRI, avgKalman, label = labels[1], color = "orange")
        plt.plot(x_axisGauge, gaugeMeasurrements, label = labels[2], color = "g")
        
        
        # Set up plot so that the dateTime strings are on the x axis as ticks
        ax = plt.axes()
        plt.xlim(np.min(x_axis), np.max(x_axis))
        
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.xaxis.set_major_locator(plt.MaxNLocator(numTicks, integer=True))
        plt.xticks(rotation = 45)
        
    
        plt.grid(True)        
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        
    else:
        
        fig, axs = plt.subplots(2, 1, sharex=True)
        
        axs[0].plot(x_axisRI, avgEM, label = labels[0], color = 'b')
        axs[0].plot(x_axisRI, avgKalman, label = labels[1], color = 'orange')
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0].grid(True)
        
        axs[1].plot(x_axisGauge, gaugeMeasurrements, label = labels[2], color = 'g')
        axs[1].set_xlim(np.min(x_axis), np.max(x_axis))
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].grid(True)
        axs[1].set_ylabel("mm")
        
        
        # Set up plot so that the dateTime strings are on the x axis as ticks
        axs[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(numTicks, integer=True))
        
        

        
    plt.xticks(rotation = 45)
    plt.xlabel("Time")
    plt.savefig(outputPath + "Rain_Intensity.pdf", bbox_inches="tight")


def plot_GOF_certainty(x_axis, GOF, threshold=0.06, outputPath = ""):
    """
    Plots the calculated Goodness-Of-Fit value against the number of frames.
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        GOF : Panda Series containing the GOF values, with dimensions [n_frames,]
        threshold : The threshold value of the GOF values used to determine whether the estimated Gaussian distribution fits the observed histogram
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf where the GOF value is plotted agains the frame number, with a superimposed line at the threshold value
    """
    
    gofValues = GOF.values
    
    plt.figure(5,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    plt.plot(x_axis, gofValues, label = GOF.name)
    
    horiz_line_data = np.array([threshold for i in range(len(x_axis))])
    plt.plot(x_axis, horiz_line_data, 'r--', label = "Threshold: {:.2f}".format(threshold)) 
    plt.grid(True)
    plt.xlabel("Frame")
    plt.xlim(np.min(x_axis), np.max(x_axis))
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(outputPath +  "Goodness_Of_Fit.pdf", bbox_inches="tight")



def analyseBossuCSVData(args):
    """
    Analyses the output of the BossuRainGauge C++ project. Creates plots of the estimated intensities over time, compared to ground truth, as wel las the state of the estimated parameters

    Input:
        args:
            - dataFilePath: Path to the data file
            - outputPath: Path to the main output folder
            - gaugeFilePath: Path to the rain gauge data file
            - latitude: Latitude of the camera
            - longitude: Longitude of the camera
            - startDateTime: Start time of the video (DD-MM-YYY HH:MM:SS)
            - frameRate: Frame rate of the camera

    Output:
        rain_dataframe: Pandas dataframe containing the input csv file
    """    
    
    ###### LOAD BOSSU OUTPUT ######
    #Load the supplied csv file
    rain_dataframe = pd.read_csv(args["dataFilePath"], sep=";")
    
    # Load the differnet parts of the csv file
    SettingsFile = rain_dataframe[rain_dataframe.columns[0]].values[0]
    InputVideo = rain_dataframe[rain_dataframe.columns[1]].values[0]
    Frames = rain_dataframe[rain_dataframe.columns[2]]
    gaussParams = rain_dataframe[rain_dataframe.columns[3:6]]
    GOF = rain_dataframe[rain_dataframe.columns[6]]
    kalmanGaussParams = rain_dataframe[rain_dataframe.columns[7:10]]
    rainIntensities = rain_dataframe[rain_dataframe.columns[10:12]]
    rainDetection = rain_dataframe[rain_dataframe.columns[12:14]]
    
    
    #If a settingsFile is found then set the threshold values accordingly
    gofThresh = 0.06
    kalmanGaussSurfaceThresh = 0.35
    if type(SettingsFile) == str and SettingsFile != "":
        settings = read_yaml(SettingsFile)
        kalmanGaussSurfaceThresh = settings["minimumGaussianSurface"]
        gofThresh = settings["maxGoFDifference"]
    
    
    ###### LOAD/PREPARE RAIN GAUGE DATA ######
    gauges = RG.RainGauges(args["gaugeFilePath"])
    FPS = args["frameRate"]
    FPM = FPS * 60
    
    rain_intensity_mod = rainIntensities["Rain Intensity"].size % FPM
    avg_EM_rain_intensity  = np.mean(np.reshape(rainIntensities["Rain Intensity"][:-rain_intensity_mod], (-1,FPM),), axis=1)
    avg_kalman_rain_intensity  = np.mean(np.reshape(rainIntensities["Kalman Rain Intensity"][:-rain_intensity_mod], (-1,FPM),), axis=1)
    minutesVideo = int(avg_EM_rain_intensity.shape[0])
    
    
    # Find the nearest rain gauge and get the datetime stamps and rain measurements
    lat = args["latitude"]
    long = args["longitude"]
    video_name = "Hjorringvej-1"
    startDateTime = datetime.datetime.strptime(args["startDateTime"], "%d-%m-%Y %H:%M:%S") + datetime.timedelta(minutes=1)
    endDateTime = startDateTime + datetime.timedelta(minutes=minutesVideo-1)    
    
    # Determine datetime values for the video rain intensity values
    videoTimeStamps = [startDateTime + datetime.timedelta(minutes=i) for i in range(minutesVideo)]
    
    
    # Get the rain measurements from the closest rain gauge
    location = RG.Location(lat, long, video_name, 0)
    print("Start Time: {}\nEnd Time: {}\nFrames analyzed: {}\nFPM: {}\nMinutes: {}".format(startDateTime, endDateTime, rainIntensities["Rain Intensity"].size, FPM, minutesVideo))
    measurement = gauges.getNearestRainData(location, startDateTime, endDateTime)
    
    gaugeTimeStamps = videoTimeStamps[0]
    rainGauge_Values = 0
    
    if measurement.perSecond:
        gaugeTimeStamps = list(measurement.perSecond.keys())
        rainGauge_Values = np.asarray(list(measurement.perSecond.values()))
   
    
    # Setup labels for plotting
    label_lst = list(rainIntensities)    
    label_lst.append("Rain Gauge")
    
    
    # Create the output directory
    OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
    
    if args["outputPath"] != "":
        OUTPUT_DIR = args["outputPath"]
        
        
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, InputVideo.split(".")[0] + "/")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    
    #Plot the different values
    plot_distribution_params(Frames, gaussParams, kalmanGaussParams, kalmanGaussSurfaceThresh, OUTPUT_DIR)
    plot_raw_rain_intensity(Frames, rainIntensities, OUTPUT_DIR)
    plot_GOF_certainty(Frames, GOF, gofThresh, OUTPUT_DIR)
    plot_rain_detection(Frames, rainDetection, OUTPUT_DIR)
    plot_avg_rain_intensity(videoTimeStamps, gaugeTimeStamps, avg_EM_rain_intensity, avg_kalman_rain_intensity, rainGauge_Values, label_lst, OUTPUT_DIR, numTicks = 9)
        
    
    return rain_dataframe



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes the .csv output file from the 'BossuRainGauge' C++ code as well as information needed to retrieve the measurements from the closest rain gauge, and plots the different distribution parameters, threholded values and rain intensity agains the frame count")
    ap.add_argument("-dataFilePath", "--dataFilePath", type=str, default = "D:/VAP_RainGauge/BossuRainGauge/Output/2018-04-30-12-44-00_leftView.mp4_Results_adapted.csv",
                    help="Path to the data file")
    ap.add_argument("-outputPath", "--outputPath", type=str, default = "",
                    help="Path to main output folder. If provided a folder will be made containing the output plots. Else it will be saved in a folder in where the script is placed")
    ap.add_argument("-gaugeFilePath", "--gaugeFilePath", type=str, default = "D:/VAP_RainGauge/PixelRainGauge/Aalborg_2018_Data",
                    help="Path to the rain gauge data file")
    ap.add_argument("-latitude", "--latitude", type=float, default = 57.05,
                    help="Latitude of the camera")
    ap.add_argument("-longitude", "--longitude", type=float, default = 9.95,
                    help="Longitude of the camera")
    ap.add_argument("-startDateTime", "--startDateTime", type=str, default = "30-04-2018 12:44:06",
                    help="Start time of the video of format 'DD-MM-YYYY HH:MM:SS'")
    ap.add_argument("-frameRate", "--frameRate", type=int, default = 30,
                    help="Frame rate of the recording camera")
    
    args = vars(ap.parse_args())
    
    r = analyseBossuCSVData(args)

