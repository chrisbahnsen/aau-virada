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

import numpy as np
import pandas as pd
import csv
import json
import datetime
import cv2
import argparse

import os
import RainGauges as RG


try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def saveJSON(filename, data):
    '''
    Takes a dict of dicts and saves to a JSON file

    Input:
        filename: path to the output file
        data: Dict of dict containing the label data
    '''

    with open(filename, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))
    
    
def getGaugeValues(gaugeFilePath, startDateTime, lat, long, FPS, numFrames, hourOffset):
    '''
    Retrieves the precipation data from the provided gauge file, based on the provided location data

    Input:
        gaugeFilepath: Path to the gauge file
        startDateTime: string of the format "DD-MM-YYYY HH:MM:SS"
        lat: latitude of the location of the video
        long: longitude of hte location of the video
        FPS: Frame rate of the video
        numFrames: Number of frames in the video
        hourOffset: Offset between the startDateTime and the rain gauge (subtracted from startDateTime)

    Output:
        gaugeTimeStamps: List of timestamp of the precipation values
        gaugeValues: Numpy array of precipation values
        frameOffset: Frames left of the first minute 
    '''

    # Load gauge
    gauges = RG.RainGauges(gaugeFilePath)
    
    # Find the nearest rain gauge and get the datetime stamps and rain measurements
    video_name = "Some Road"
    
    # Get start time
    startDateTime = datetime.datetime.strptime(startDateTime, "%d-%m-%Y %H:%M:%S")
    startSeconds = startDateTime.second
    startDateTime -= datetime.timedelta(seconds = startSeconds)
    startDateTime -= datetime.timedelta(hours = hourOffset)
    
    # Calculate how many frames left of the starting minute e.g. 16:00:45, has 15 seconds left 
    # This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
    # These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:00:01
    FPM = FPS * 60

    if startSeconds > 0:
        frameOffset = (60 - startSeconds) * FPS - 15
    else:
        frameOffset = 0
        
    # Determine how many minutes the video spans
    if numFrames > frameOffset:
        minutesVideo = int(np.ceil((numFrames-frameOffset)/FPM))
    else:
        minutesVideo = 0
    
    # Get the end time of the video
    endDateTime = startDateTime + datetime.timedelta(minutes=minutesVideo)    
	
    # Get the rain measurements from the closest rain gauge
    location = RG.Location(lat, long, video_name, 0)
    measurement = gauges.getNearestRainData(location, startDateTime, endDateTime)
    
    if measurement.perSecond:
        gaugeTimeStamps = list(measurement.perSecond.keys())
        gaugeValues = np.asarray(list(measurement.perSecond.values()))
    
    return gaugeTimeStamps, gaugeValues, frameOffset


def generateLabels(args):
    '''
    Takes a csv file with information about the relevant videos.
    Based on this information the precipation labels are found from the nearest rain gauge in the provided rain gauge file
    The precipation labels are saved as a JSON file in a folder 'labels' in the work dir

    Input:
        args:
            - csvFilePath: Path to the video csv file
            - outputFile: Name of the output file
            - binary: Whether to binarize the rain gauge data or not
            - vid_label: Whether to use the label provided in the input csv file for the entire video
            - precipation: Whether to use precipation data for labels
            - hour_offset: How many hours to offset the rain gauge data by
            - verbose: Whether to print information

    '''

    csv_file = args["csvFilePath"]
    df = pd.read_csv(csv_file, sep = ",")
    
    folder = "./labels"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    binary_label = args["binary"]
    vid_label = args["vid_label"]
    precipation_label = args["precipation"]
    hour_offset = args["hour_offset"]
    label_dict = {}
    
    verbose = args["verbose"]
    
    # Go through each video in the supplied csv file
    for index, row in df.iterrows():
        FPS = row["FPS"]
        gaugeFilePath = row["GaugeFilePath"]
        videoFilePath = row["VideoFilePath"]
        latitude = row["Latitude"]
        longitude = row["Longitude"]
        startDateTime = row["StartTime"]
        supplied_label = row["FileLabel"]
        if np.isfinite(supplied_label):
            supplied_label = np.int(supplied_label)
        
        if verbose:
            print("Row {}\n\tGauge File Path: {}\n\tVideoFilePath: {}\n\tStart Datetime: {}\
                  \n\tLatitude: {}\n\tLongitude: {}\n\tFPS: {}\n\tFrame Label: {}\n".format(index,
                                                                                        gaugeFilePath,
                                                                                        videoFilePath,
                                                                                        startDateTime,
                                                                                        latitude,
                                                                                        longitude,
                                                                                        FPS,
                                                                                        supplied_label))
        filename = os.path.basename(videoFilePath)
        
        if not vid_label:
            ## Uses the supplied label for the ENTIRE video, and stored as a single number
            labels = supplied_label
            frameOffset = None
            numFrames = None
            timestamps = []
        else:
            # Load video information, in order to retrieve corresponding precipation data
            # Labels are stored per minute in a list

            cap = cv2.VideoCapture(videoFilePath)
        
            if not cap.isOpened(): 
                print ("Could not open {}".format(videoFilePath))
                continue
            
            numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidFPS    = int(cap.get(cv2.CAP_PROP_FPS))
            if verbose:
                print("Video {}:\n\tFrames: {}\n\tFPS: {}\n\n".format(videoFilePath, numFrames, vidFPS))
            
            assert FPS == vidFPS, "The supplied FPS, {}, and video FPS, {}, differ".format(FPS,vidFPS)
             
            # Get the gauge values for the video
            timestamps, values, frameOffset = getGaugeValues(gaugeFilePath, startDateTime, latitude, longitude, FPS, numFrames, hour_offset)
            
            if binary_label:
                if  precipation_label:
                    ## Binarizes the rain gauge data and saves a value per minute
                    labels = (values.astype(np.bool)).astype(int)
                else:
                    ## Creates a list of the same length returned from the rain gauge data, but fills it with the supplied label
                    labels = np.ones(len(values),dtype=np.int)*supplied_label
            else:
                ## Uses the direct rain gauge data as labels per minute
                labels = values

            # convert numpy array to a list
            labels = labels.tolist()
        
        # Save video label dict into dict
        label_dict[filename] = {"labels": labels,
                                "frameOffset": frameOffset,
                                "timestamps": [x.strftime("%Y-%m-%d %H:%M:%S") for x in timestamps],
                                "frameCount": numFrames,
                                "FPM": FPS * 60}
        if verbose:
            print()
            print(filename, numFrames, vidFPS)
            print(label_dict[filename])
    
    # save dict to JSON
    saveJSON(os.path.join(folder,args["outputFile"]), label_dict)

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Generates precipation labels for each video in the provided csv video, based on data from the neasrest rain gauge")
    ap.add_argument("-csvFilePath", "--csvFilePath", type=str, default="labelsCSV.csv",
                    help="Path to the csv file containing information per video")
    ap.add_argument("-outputFile", "--outputFile", type=str, default = "labels.json",
                    help="Filename for the output JSON file. Saved in the active dir in a subdir called 'labels'")
    ap.add_argument('--binary', action="store_true",
                    help='Use binary labels? If not set, continous labels are generated')
    ap.add_argument('--precipation', action="store_true",
                    help='Use precipation data from rain guage for labels')
    ap.add_argument('--vid_label', action="store_false",
                    help='Use specific label per frame/minute?')
    ap.add_argument('--hour_offset', type=int, default=2,
                    help='How many hours to offset the raing guage data')
    ap.add_argument('--verbose', action="store_true",
                    help='Whether information should be printed')
    args = vars(ap.parse_args())
    generateLabels(args)