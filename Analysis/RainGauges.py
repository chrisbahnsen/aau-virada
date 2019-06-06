# MIT License
# 
# Copyright(c) 2017 Aalborg University
# Chris H. Bahnsen, June 2017
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

import datetime
from collections import namedtuple
import math
import os

def monthToInt(monthName):
    return {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12,
        }.get(monthName, 0) # Invalid month is default


def toRadians(number):
    return number * math.pi / 180

class RainMeasurement(object):

    def __init__(self, location):
        self.location = location
        self.totalInMm = 0.
        self.perSecond = {}

class Location(object):
    """Represents a physical location, given by the latitude
       and longitude, WGS84
    """

    def __init__(self, lat, long, name, id):
        self.lat = lat
        self.long = long
        self.name = name
        self.id = id

    def measureDistance(self, location):
        # Code converted from JavaScript, original source
        # http://www.movable-type.co.uk/scripts/latlong.html

        R = 6371e3; # metres
        phi1 = toRadians(self.lat)
        phi2 = toRadians(location.lat)
        deltaPhi = toRadians(location.lat-self.lat)
        deltaLambda = toRadians(location.long-self.long)

        a = (math.sin(deltaPhi/2) * math.sin(deltaPhi/2) +
            math.cos(phi1) * math.cos(phi2) *
            math.sin(deltaLambda/2) * math.sin(deltaLambda/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        d = R * c

        return d

class Recording(object):
    """Represents a rain gauge recording with a
       physical location and start and end time
    """

    def __init__(self, location, startTime, endTime, fileName, columnIndex):
        self.location = location
        self.startTime = startTime
        self.endTime = endTime
        self.fileName = fileName
        self.columnIndex = columnIndex

    def getPrecipitation(self, startTime, endTime):
        """ Returns a PrecipitationMeasurements tuple of the rainfall
            between the startTime and endTime if they are 
        
        """
        if startTime < self.startTime:
            return None
        if endTime > self.endTime:
            return None
        
        measurements = RainMeasurement(self.location)

        with open(self.fileName) as f:
            for line in f:
                entries = line.split(',')

                if len(entries) > self.columnIndex:
                    dateTimeSplit = entries[0].split(' ')
                    
                    if len(dateTimeSplit) == 2:
                        dateSplit = dateTimeSplit[0].split('-')
                        timeSplit = dateTimeSplit[1].split(':')
                        
                        if len(dateSplit) == 3 and len(timeSplit) == 3:
                            dateTime = datetime.datetime(int(dateSplit[0]),
                                        int(dateSplit[1]), int(dateSplit[2]),
                                        int(timeSplit[0]), int(timeSplit[1]), 
                                        int(timeSplit[2]))

                            if dateTime >= startTime and dateTime <= endTime:
                                measurements.totalInMm += float(entries[self.columnIndex])

                                measurements.perSecond[dateTime] = float(entries[self.columnIndex])

        return measurements
                               

class RainGauges(object):
    """Manages a list of data from real rain gauges 
       and provides an interface for retrieving the rain 
       data from a particular location and time span
        
    """

    def __inspectMeasurementFiles(self):
        """ Inspect the rain measurements in the folder provided in the 
            initialisation and create an object that provides easy lookup
            of the location and time span of the measurements"""

        files = [file for file in os.listdir(self.rainGaugeFolder)
                 if os.path.isfile(os.path.join(self.rainGaugeFolder, file))]


        # Get information of the location of the rain gauges
        for file in files:
            if 'GaugeInfo' in file:
                with open(os.path.join(self.rainGaugeFolder, file)) as f:
                    lines = f.readlines()

                    for idx in range(1, len(lines)):
                        entries = lines[idx].split(',')

                        if len(entries) < 8:
                            continue

                        id = int(entries[0])
                        name = entries[1]
                        lat = float(entries[6])
                        long = float(entries[7].replace('\n',''))

                        gauge = Location(lat, long, name, id)
                        self.rainGaugeLocations[id] = gauge

        
        # Get the information of the duration of the rain gauge recordings 
        # and couple the location of the gauge with the gauge id
        for file in files:
            if '.txt' in file and 'GaugeInfo' not in file:
                fileInfoParts = file.split('-')

                if len(fileInfoParts) >= 7:
                    startTime = datetime.datetime(int(fileInfoParts[3]), monthToInt(fileInfoParts[2]), int(fileInfoParts[1]))
                    endTime = datetime.datetime(int(fileInfoParts[6].split('.')[0]), monthToInt(fileInfoParts[5]), int(fileInfoParts[4]))

                    with open(os.path.join(self.rainGaugeFolder, file)) as f:
                        # Just read the first line - it contains the information
                        # we need for now
                        entries = f.readline().split(',')

                        if len(entries) > 1:
                            for idx in range(1, len(entries)):
                                location = self.rainGaugeLocations[id]

                                recording = Recording(location, startTime, endTime, 
                                            os.path.join(self.rainGaugeFolder, file), idx)  
                                self.rainGaugeRecordings.append(recording)

    
    def getNearestRainData(self, location, startTime, endTime):

        # Find the closest rain gauge to the location listed
        # In order to quality, the rain gauge must have a recording
        # within the specified start and end time

        shortestDistance = 10000 # If we can't find a rain gauge within 10 km, we have failed
        bestRecording = None

        for recording in self.rainGaugeRecordings:

            distance = recording.location.measureDistance(location)
            if (distance < shortestDistance 
                and recording.startTime <= startTime 
                and recording.endTime >= endTime):
                shortestDistance = distance
                bestRecording = recording

        return bestRecording.getPrecipitation(startTime, endTime)

    def __init__(self, rainGaugeFolder):
        
        self.rainGaugeFolder = rainGaugeFolder
        self.rainGauges = {}

        self.nearestRainGauge = None

        self.rainGaugeRecordings = []
        self.rainGaugeLocations = {}

        self.__inspectMeasurementFiles()