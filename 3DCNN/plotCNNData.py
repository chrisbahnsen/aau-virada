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
import pickle as pkl
import matplotlib.pyplot as plt
import argparse

def plotGraph(trn_y= None, val_y= None, x_axis= None, y_axis= None, title= None, output= None):
    """
    Plots a line plot
    """

    plt.figure()
    plt.clf()
    t = np.arange(1, len(trn_y)+1)
    print(len(t))
    print(len(trn_y))

    # red dashes, blue squares and green triangles
    if trn_y:
        plt.plot(t, trn_y, "b",label="Train")
    if val_y:
        plt.plot(t, val_y, "r", label="Validation")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output) 


def plotCNNData(args):
    """
    Reads a pkl file containing the different stored data saaved with Tensorbaord, fro ma CNN training session

    Plots the training/validation accuracy, loss, time and learning rate

    Input:
        args:
            - inputFile: Path to input file
    """

    with open(args["inputFile"],'rb') as f:
        x = pkl.load(f)
        print(x.keys())


    plotGraph(x["TRN_Acc"], x["VAL_Acc"], "Epoch", "Accuracy (%)", "", "accuracy.pdf")
    plotGraph(x["TRN_Loss"], None, "Iteration", "Loss", "", "loss.pdf")
    plotGraph(x["TRN_AVG_Loss"], x["VAL_AVG_Loss"], "Epoch", "Loss", "", "avg_loss.pdf")
    plotGraph(x["TRN_Time"], None, "Iteration", "Time (s)", "", "time.pdf")
    plotGraph(x["TRN_AVG_Time"], x["VAL_AVG_Time"], "Epoch", "Time (s)", "", "avg_time.pdf")
    plotGraph(x["Learning_Rate"], None, "Epoch", "LR", "", "lr.pdf")


    
if __name__ == "__main__":   
    ap = argparse.ArgumentParser(
            description = "Takes a pkl file containing the Tensorboard information from trainining session")
    ap.add_argument("-inputFile", type=str, default="eventData.pkl",
                    help="Path to the input file")
 
    args = vars(ap.parse_args())
    plotCNNData(args)