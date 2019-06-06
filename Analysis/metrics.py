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
from sklearn.metrics import auc

def calculate_type_errors(observations, labels):
    '''
    Calculates the different type errors

    Input:
        observations: Numpy array containing the predicted values
        labels: Numpy array containing the actual values
    
    Output:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives
    '''

    obs_not = np.logical_not(observations)
    label_not = np.logical_not(labels)
    
    ## Uses the boolean and operator ( & ) so only indecies where both are True are returned
    TP = len((np.where(labels & observations))[0])
    TN = len((np.where(label_not & obs_not))[0])
    FP = len((np.where(label_not & observations))[0])
    FN = len((np.where(labels & obs_not))[0])
    
    return TP, TN, FP, FN    


def calculate_accuracy(TP, TN, FP, FN):
    """
    Calculates the prediction accuracy of the supplied reference and test data
    
    Input:
        ref: Numpy boolean array of the reference data, of size [n_samples, ]
        test: Numpy boolean array of the test data, of size [n_samples, ]
        
    Output:
        accuracy: The accuracy of the predictions in the test data
    """
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return accuracy
    

def calculate_rates(TP, TN, FP, FN):
    """
    Calculates the True and False Positive and Negative rates of the supplied reference and test data
    
    For TPR and TNR, if the denominator is 0, the result is defined as 1, as there were no false positive/negatives
    
    Input:
        ref: Numpy boolean array of the reference data, of size [n_samples, ]
        test: Numpy boolean array of the test data, of size [n_samples, ]
        
    Output:
        TPR: The True Positive Rate (also known as Sensitivity)
        TNR: The True Negative Rate (also known as Specificity)
        FPR: The False Positive Rate (equal to 1 - Specificity)
        FNR: The False Negative Rate (equal to 1 - Sensitivity)
    """
    
    
    if (TP + FN) > 0:
        TPR = TP / (TP + FN)  # Sensitivity
    else: 
        TPR = 1
    
    if (TN + FP) > 0:
        TNR = TN / (TN + FP) # Specificity
    else:
        TNR = 1
        
    FPR = 1 - TNR
    FNR = 1 - TPR
    
    return TPR, TNR, FPR, FNR


def calculate_predictive_value(TP, TN, FP, FN):
    """
    Calculates the Positive and Negative Predictive value of the supplied reference and test data
    
    For PPV and NPV, if the denominator is 0, the result is defined as 1, as there were no false positive/negatives
    
    Input:
        ref: Numpy boolean array of the reference data, of size [n_samples, ]
        test: Numpy boolean array of the test data, of size [n_samples, ]
        
    Output:
        PPV: The Positive Predictive Value
        NPV: The Negative Predictive Value 
    """
    
    
    if (TP + FP) > 0:
        PPV = TP / (TP + FP)
    else: 
        PPV = 1
    
    if (TN + FN) > 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 1
        
    
    return PPV, NPV


def calculate_precision_recall(TP, TN, FP, FN):
    """
    Calculates the Precision and Recall meassure of the supplied reference and test data, as per https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
    Incase of the denominator being zero for precision/recall, the values are set to 1, as per https://stats.stackexchange.com/a/16242 and https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    
    Input:
        ref: Numpy boolean array of the reference data, of size [n_samples, ]
        test: Numpy boolean array of the test data, of size [n_samples, ]
        
    Output:
        precision: The precision value of the supplied data
        recall: The recall value of the supplied data
    """
    
    if (TP+FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 1
        
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 1
        
    return precision, recall
    

def calculate_F_score(precision, recall, beta = None):
    """
    Calculates the F-meassure of the supplied data, using the equation as per https://en.wikipedia.org/wiki/Precision_and_recall#F-measure :
        F = (1 + beta^2) * (precision * recall)/(beta^2 * precision + recall)
        
    If the recall and precision is 0, the f-score is set to 0, as no correct true positives are in the analyzed data
    
    Input:
        precision: The precision value of the data
        recall: The recall value of the data
        beta: A weight parameter of the F-Meassure. Default is None, which defaults to the F1 meassure. Should only be set to float values
        
    Output:
        F: The F meassure value of the supplied valies
    """
    
    if precision is None:
        return None
    if recall is None:
        return None
    if (precision + recall) == 0:
        return 0
    elif beta is None:
        F = 2 * (precision * recall) / (precision + recall)
    else:
        F = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return F


def calculate_informedness(TPR, TNR):
    """
    Calculates the informedness/Youden's J meassure of the supplied data, using the equation as per: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        Informedness = TPR+TNR-1
            
    Input:
        TPR: True Positive Rate
        TNR: True Negative Rate

    Output:
        Informedness
    """
    
    return TPR+TNR-1


def calculate_markedness(PPV, NPV):
    """
    Calculates the markedness meassure of the supplied data, using the equation as per: http://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf (page 5)
        Markedness = PPV+NPV-1
            
    Input:
        PPV: Positive Prediction Value
        NPV: Negative Pediction Value

    Output:
        Markedness
    """
    
    return PPV+NPV-1


def calculate_MCC(TP, TN, FP, FN):
    """
    Calculates Matthews Correlation Coefficient of the supplied data, using the equation as per: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        MCC = TP*TN-FP*FN/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        
    MCC occupies the range [-1; 1], with 1 being perfect prediction, 0 being totally random, and, -1 being totally incorrect
    If any of the sums inthe denominator is 0, the denominator is set to 1, leading to an MCC of 0
    
    Input:
        TP: True positives
        TN: True negatives
        FP: False positives
        FN: False negatives

    Output:
        MCC
    """
    
    numerator = TP*TN-FP*FN
    denominator = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    if denominator == 0:
        denominator = 1
    
    return numerator/np.sqrt(denominator)


def calculate_AUC(x_axis, y_axis):
    """
    Calculates the Area Under Curve (AUC) for the supplied x/y values. 
    It is assumed that the x axis data is either monotonoically increasing or decreasing
    
    Input:
        x_axis: List/numpy array of values
        y_axis: list/numpy array of values

    Output:
        AUC
    """
    
    return auc(x_axis, y_axis)


def calculate_classification_metrics(metric_dict, observation, labels, f_beta = None):
    """
    Takes a set of observations and labels from which the different binary classification metrics are computed. These are saved in the provided dict.

    Input:
        metric_dict: Dictionary containing lists for each metric
        observations: List/Numpy array of the observations
        labels: List/Numpy array of the labels
        f_beta = Parameter for the F-score. If None, F1 is calcualted

    Output:
        metric_dict: The updated input dictionary.
    """

    type_errors = calculate_type_errors(observation, labels)
    type_error_rates = calculate_rates(*type_errors)
    predictive_values = calculate_predictive_value(*type_errors)
    accuracy = calculate_accuracy(*type_errors)
    prec_recall = calculate_precision_recall(*type_errors)
    f_score = calculate_F_score(*prec_recall, f_beta)
    inv_f_score = calculate_F_score(predictive_values[1], type_error_rates[1], f_beta)
    informedness =  calculate_informedness(type_error_rates[0], type_error_rates[1])
    markedness = calculate_markedness(*predictive_values)
    MCC = calculate_MCC(*type_errors)
    
    metric_dict["Type Errors"].append(type_errors)
    metric_dict["Type Rates"].append(type_error_rates)
    metric_dict["Predictive Values"].append(predictive_values)
    metric_dict["Accuracy"].append(accuracy)
    metric_dict["Precision-Recall"].append(prec_recall)
    metric_dict["F1-score"].append((f_score, inv_f_score))
    metric_dict["Informedness"].append(informedness)
    metric_dict["Markedness"].append(markedness)
    metric_dict["MCC"].append(MCC)
    
    return metric_dict
   
    
def calculate_classification_metrics_full_dataset(observations, thresholds = None):
    """
    Giving several observations (different files), which have been analyzed at different thresholds (for the per frame analyses) calcualte the overall performance for the entire dataset

    Input:
        observations: List containing the accumulated type errors at different thresholds
        thresholds: The different thresholds used for the analyses

    Output:
        metric_dict: Dictionary containing all the different binary classification metrics from the provided type errors
    """

    metric_dict = {"Type Errors": [],
                        "Type Rates": [],
                        "Predictive Values": [],
                        "Accuracy": [],
                        "Precision-Recall": [],
                        "F1-score": [],
                        "Informedness": [],
                        "Markedness": [],
                        "MCC": []}
    metric_dict["Type Errors"] = observations
    
    if thresholds:
        for i, _ in enumerate(thresholds):
            type_errors = metric_dict["Type Errors"][i]
            metric_dict["Type Rates"].append(calculate_rates(*type_errors))
            metric_dict["Predictive Values"].append(calculate_predictive_value(*type_errors))
            metric_dict["Accuracy"].append(calculate_accuracy(*type_errors))
            metric_dict["Precision-Recall"].append(calculate_precision_recall(*type_errors))
            metric_dict["F1-score"].append((calculate_F_score(*metric_dict["Precision-Recall"][-1]), calculate_F_score(metric_dict["Predictive Values"][-1][1], metric_dict["Type Rates"][-1][1])))
            metric_dict["Informedness"].append(calculate_informedness(metric_dict["Type Rates"][-1][0], metric_dict["Type Rates"][-1][1]))
            metric_dict["Markedness"].append(calculate_markedness(*metric_dict["Predictive Values"][-1]))
            metric_dict["MCC"].append(calculate_MCC(*type_errors))

        TPR = [x[0] for x in metric_dict["Type Rates"]]
        FPR = [x[2] for x in metric_dict["Type Rates"]]
        Precision = [x[0] for x in metric_dict["Precision-Recall"]]
        Recall = [x[1] for x in metric_dict["Precision-Recall"]]
        metric_dict["AUROC"] = calculate_AUC(FPR, TPR)
        metric_dict["AUPR"] = calculate_AUC(Recall, Precision)
    else:
        metric_dict["Type Errors"] = observations
        type_errors = metric_dict["Type Errors"][0]
        metric_dict["Type Rates"].append(calculate_rates(*type_errors))
        metric_dict["Predictive Values"].append(calculate_predictive_value(*type_errors))
        metric_dict["Accuracy"].append(calculate_accuracy(*type_errors))
        metric_dict["Precision-Recall"].append(calculate_precision_recall(*type_errors))
        metric_dict["F1-score"].append((calculate_F_score(*metric_dict["Precision-Recall"][-1]), calculate_F_score(metric_dict["Predictive Values"][-1][1], metric_dict["Type Rates"][-1][1])))
        metric_dict["Informedness"].append(calculate_informedness(metric_dict["Type Rates"][-1][0], metric_dict["Type Rates"][-1][1]))
        metric_dict["Markedness"].append(calculate_markedness(*metric_dict["Predictive Values"][-1]))
        metric_dict["MCC"].append(calculate_MCC(*type_errors))

    return metric_dict