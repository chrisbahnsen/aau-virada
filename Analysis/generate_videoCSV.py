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
import csv
import argparse


def Crossing1Files(fileName, ind=None):
    """
    Takes a filename of the structure used for the Crossing 1 videos - YYYMMDD_hhmmss-suffix.mkv
    Retrieves the datetime of the start of the video

    Input:
        fileName: Name of the file
        ind: Indicator of whether it is the left or right crop of the video which is wanted. Optional

    Output:
        datetime: String of the format  DD-MM-YYYY hh:mm:ss
    """

    date = fileName[:8]
    time = fileName[9:15]
    
    if ind:
        if ind == "left":
            if "right" in fileName:
                return None
        elif ind == "right":
            if "left" in fileName:
                return None
    
    datetime = "{}-{}-{} {}:{}:{}".format(date[6:8], date[4:6], date[:4], time[:2], time[2:4], time[4:6])
    
    return datetime
    

def Crossing2Files(fileName):
    """
    Takes a filename of the structure used for the Crossing 2 videos - M-DD-hh-suffix.mkv
    Retrieves the datetime of the start of the video

    Currently hardcoded to encode datetime with the year 2013, and assumes the video starts at the exact hour

    Input:
        fileName: Name of the file

    Output:
        datetime: String of the format  DD-MM-YYYY hh:mm:ss
    """
    dateTimeStr = fileName.split('-')

    if len(dateTimeStr) > 2:
        year = 2013
        month = int(dateTimeStr[0])
        day = int(dateTimeStr[1])
        hour = int(dateTimeStr[2])
        if dateTimeStr[3].isdigit():
            minute = int(dateTimeStr[3])
            sec = int(dateTimeStr[4])
        else:
            minute = 0
            sec = 0

        dateTime = "{}-{}-{} {:02d}:{:02d}:{:02d}".format(day, month, year, hour, minute, sec)
        return dateTime
    else:
        return None


def generate_csv(args):
    '''
    Takes path to a set of videos, and construct a csv file which can be used for input to the Bossu Video analysis scripts

    Input:
        args:
            - videoDirPath: Path to the vidoe folder
            - gaugeFilePath: Path to the gauge file
            - outputFile: Path to the output csv file
            - ind: Indicator of whether to use the left or right crop (For Crossing1 videos)
            - FPS: FPS of the videos
            - latitude: Latitude of the camera position
            - longitude: Longitude of the camera position
            - crossing: Whether video filenames adhere to Crossing1 or Crossing2 format

    '''

    videoDirPath = args["videoDirPath"]
    gaugeFilePath = args["gaugeFilePath"]
    outputFile = args["outputFile"]
    
    ind = args["ind"]
    FPS = args["FPS"]
    latitude = args["latitude"]
    longitude = args["longitude"]
    
    crossing = args["crossing"]
    
    ind = ind.lower()
    
    
    with open(outputFile, "w", newline='') as outFile:
        writer = csv.writer(outFile)
        
        firstRow = []
        firstRow.append("VideoFilePath")
        firstRow.append("GaugeFilePath")
        firstRow.append("Latitude")
        firstRow.append("Longitude")
        firstRow.append("StartTime")
        firstRow.append("FPS")
        firstRow.append("FileLabel")
        
        writer.writerow(firstRow)
        
        for file in os.listdir(videoDirPath):

            if not any(x in str(file) for x in ['.mp4', ".mkv", ".avi"]):
                continue
            
            if crossing == 1:
                datetime = Crossing1Files(file, ind)
            elif crossing == 2:
                datetime = Crossing2Files(file)
            else:
                raise ValueError("Provided parameter 'crossing' with value {} - No method implemented for this".format(crossing))
            
            if datetime == None:
                continue
            
            row = []
            row.append(os.path.join(videoDirPath, file))
            row.append(gaugeFilePath)
            row.append(latitude)
            row.append(longitude)
            row.append(datetime)
            row.append(FPS)
            row.append(None)
            
            
            writer.writerow(row)
        
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Creates a csv file for a set of videos, containing relevant meta information")
    ap.add_argument("-videoDirPath", type=str, default="U:/videosurv/Aalborg-Skibsbyggerivej/EncodedCRF",
                    help="Path to the folder containing the videos of interest")
    ap.add_argument("-outputFile", type=str, default = "labelsCSV_2hour.csv",
                    help="Filename for the output csv file. Saved in the work dir")
    ap.add_argument('-gaugeFilePath', type=str, default = "D:/VAP_RainGauge/PixelRainGauge/Aalborg_2018_Data-Corrected-2hour",
                    help='Path to dir containing rain gauge information')
    ap.add_argument("--ind", type=str, default="left",
                    help="Indicate whether only left or right side should be used. if neither 'left' or 'right' all files will be used")
    ap.add_argument("--FPS", type=int, default=30,
                    help="Frame rate of video")
    ap.add_argument("--latitude", type=float, default=57.05,
                    help="Latitude of camera")
    ap.add_argument("--longitude", type=float, default=9.95,
                    help="Longitude of camera")
    ap.add_argument("--crossing", type=int, default=1,
                    help="Whether the provided videos have filenames structured according to crossing 1 or crossing 2")
    
    args = vars(ap.parse_args())
    generate_csv(args)