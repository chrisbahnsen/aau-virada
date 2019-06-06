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

import tensorflow as tf
import pickle as pkl
import argparse
print("Tensorflow version".format(tf.__version__))

def read_tfEvents(args):
    """
    Reads the output Tensorboard file from a CNN training sessions, and saves it into a pickle file

    Input:
        args:
            - inputFile: Path to the input file
            - outputFile: Path to the output file
    """

    summary_path = args["inputFile"]

    training_loss = []
    training_time = []
    training_avg_loss = []
    training_avg_time = []
    training_acc = []

    validation_avg_loss = []
    validation_avg_time = []
    validation_acc = []

    learning_rate = []

    config = []

    for e in tf.train.summary_iterator(summary_path):
        for v in e.summary.value:
            if v.tag == 'Loss/Training':
                training_loss.append(v.simple_value)
            elif v.tag == "Time/Training":
                training_time.append(v.simple_value)
            elif v.tag == "Loss/Training-Avg":
                training_avg_loss.append(v.simple_value)
            elif v.tag == "Time/Training-Avg":
                training_avg_time.append(v.simple_value)
            elif v.tag == "Accuracy/Training":
                training_acc.append(v.simple_value)
                
            elif v.tag == "Loss/Validation-Avg":
                validation_avg_loss.append(v.simple_value)
            elif v.tag == "Time/Validation-Avg":
                validation_avg_time.append(v.simple_value)
            elif v.tag == "Accuracy/Validation":
                validation_acc.append(v.simple_value)
            
            elif v.tag == "Learning_Rate":
                learning_rate.append(v.simple_value)

            elif v.tag == "config/text_summary":
                config.append(v)
            else:
                print(v.tag)
                print(v.simple_value)

    print(len(validation_acc), validation_acc)
    print(len(training_acc), training_acc)
    print(len(training_loss))
    print(config)

    with open(args["outputFile"],'wb') as f:
        outDict = {"TRN_Loss": training_loss,
                "TRN_Time": training_time,
                "TRN_AVG_Loss":training_avg_loss,
                "TRN_AVG_Time": training_avg_time,
                "TRN_Acc": training_acc,
                "VAL_AVG_Loss": validation_avg_loss,
                "VAL_AVG_Time": validation_avg_time,
                "VAL_Acc": validation_acc,
                "Learning_Rate": learning_rate}
        pkl.dump(outDict, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Creates a csv file for a set of videos, containing relevant meta information")
    ap.add_argument("-inputFile", type=str, default="D:/TraficData_results/14Apr2019/Apr08_07-30-09_joha/events.out.tfevents.1554708609.joha",
                    help="Path to the folder  file")
    ap.add_argument("-outputFile", type=str, default = "eventData.pkl",
                    help="Filename for the output pkl file. Saved in the work dir")

    args = vars(ap.parse_args())
    read_tfEvents(args)