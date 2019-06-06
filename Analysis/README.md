Scripts used for analysing the 3DCNN and Bossu rain detection algorithm
=====

## Generate labels

First a CSV containing all necessary information per video needs to be constructed. This is done by using the *generate_videoCSV.py* script. This script takes a directory with containing all videos as input, as well as FPS, latitude, longitude, filepath to file containing the rain gauge information. This information is assumed to be similar for all videos in the directory. Furthermore, it is assumes the filenames are either of format *YYYMMDD_hhmmss-suffix.mkv* or *M-DD-hh-suffix.mkv*.

In order to generate labels as used for the 3DCNN and the result analysis, the *generate_labels.py* script is used. The script requries a csv file from the *generate_videCSV.py* script. Furthermore the *RainGauges.py* script is internally used, using the gauge file which is provided per video file, in the aforementioned csv file.

## Analysze dataset

In order to analyze a dataset the *analyseDataset.py* script can be used. The script requries a csv file with all the relevant video filenames as well as a corresponding JSON labels file.
The occurence rate of rain/no rate, the total amount of frames and more is then calculated.


## Evaluate rain detectors

In order to analyze the output of the rain detectors, there are two separate scripts. *evaluateBossu.py* is used for the output of the Bossu et al. algorithm, while *evaluate3DCNN.py* is used for the output of the 3DCNN algorithm.