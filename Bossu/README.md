Rain estimation using Bossu et al.
=====

This code base is for estimating whether rain is present in a single image. It is a reimplmentation of the algorithm outlined in J. Bossu, N. Hautière, and J.P. Tarel in "Rain or snow detection in image sequences through use of a histogram of orientation of streaks." appearing in International Journal of Computer Vision, 2011

## C++ code

The algorithm is implemeted in C++ using OpenCV version 3.1.0. The code has been as faithfully implemented as possible. When guesses or liberties have been made, this is clearly noted in the source code.

## Paramater search

In order to run the algorithm over several different parameters (or just a single combination) the *BossuVideoAnalysis.py* script can be used. This script takes the csv output of the *generate_videoCSV.py* script in the **Analysis** folder, as well as several values per parameter. After each video evlauation the *BossuCSVAnalysis.py* script is called, and produces some explanatory pdfs in order to see the state determined distribution parameters over time and more.

The kalman filter parameters can be optimized without evaluating on the videos at all time. This is done by using the *gofAndSurfaceSweep.py* script, which iterates the Goodness-of-Fit and Surface threshold values provided. The results of this test can be summarized using the *analyzeParameterSweep.py* script.



