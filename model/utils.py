import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
import tensorflow as tf
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import os
import sys
import pickle
from data import DataSet, DataSet2, DataSetWithMask, DataSetWithMask2
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index
RESULT_SAVE_DIR = 'J:/实验室/survival-analysis_code20.10.29/model/result'
DATA_SAVE_DIR = 'j:/实验室/survival-analysis_code20.10.29/model/test/data'


def partial_log_likelihood(prediction, t, y):
    """
    calculate cox loss
    :param prediction: prediction of model
    :param t: event happen at the 't'th day
    :param y: true label
    :return:
    """
    risk = tf.reshape(prediction, [-1])
    time = tf.reshape(t, [-1])
    E = tf.reshape(y, [-1])
    sort_idx = tf.argsort(time, direction='DESCENDING')
    E = tf.gather(E, sort_idx)
    risk = tf.gather(risk, sort_idx)
    hazard_ratio = tf.exp(risk)
    log_risk = tf.math.log(tf.cumsum(hazard_ratio))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = tf.multiply(uncensored_likelihood, E)
    neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.01
    return neg_likelihood


def calculate_score(y_label, y_prediction, print_flag=False):
    """
    :param y_label: true label
    :param y_prediction: prediction of model
    :param print_flag: print pr not
    :return: auc, precision, recall, f_score, accuracy
    """
    try:
        auc = roc_auc_score(y_label, y_prediction)
        fpr, tpr, thresholds = roc_curve(y_label, y_prediction)
        threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_label = (y_prediction >= threshold) * 1
        precision = precision_score(y_label, y_pred_label)
        recall = recall_score(y_label, y_pred_label)
        f_score = f1_score(y_label, y_pred_label)
        accuracy = accuracy_score(y_label, y_pred_label)
        if print_flag:
            print('auc:{} precision:{} recall:{} f_score:{} accuracy:{}'.format(auc, precision, recall, f_score,
                                                                                accuracy))
    except:
        return 0,0,0,0,0
    return y_pred_label, auc, precision, recall, f_score, accuracy



def dataset_spilt_and_store(data, k_folds, data_type, store_name):
    """
    :param data: 待划分的数据集
    :param k_folds: 划分比例
    :param data_type:  数据集类型 （0：没有相对时间的数据集； 1：有相对时间的数据集）
    :param store_name: 划分好的数据集存储名称
    :return: null
    """
    test_size = 1 / k_folds
    train_size = 1 - test_size
    if data_type == 0:
        features, labels, time = data
        features_index = features.reshape(features.shape[0], -1)
        label_index = labels.reshape(labels.shape[0], -1)
        split = StratifiedShuffleSplit(k_folds, test_size, train_size, 1).split(features_index, label_index)
        train_index, test_index = next(split)
        train_set = DataSet(features[train_index], time[train_index], labels[train_index])
        test_set = DataSet(features[test_index], time[test_index], labels[test_index])
    else:
        features, labels, time, day = data
        features_index = features.reshape(features.shape[0], -1)
        label_index = labels.reshape(labels.shape[0], -1)
        split = StratifiedShuffleSplit(k_folds, test_size, train_size, 1).split(features_index, label_index)
        train_index, test_index = next(split)
        train_set = DataSet2(features[train_index], time[train_index], labels[train_index], day[train_index])
        test_set = DataSet2(features[test_index], time[test_index], labels[test_index], day[test_index])

    with open('{}/train_set_{}{}.pkl'.format(DATA_SAVE_DIR, store_name, data_type), 'wb') as f:
        pickle.dump(train_set, f)
    with open('{}/test_set_{}{}.pkl'.format(DATA_SAVE_DIR, store_name, data_type), 'wb') as f:
        pickle.dump(test_set, f)


def get_true_data_mask(input_x):
    input_shape = input_x.shape
    mask = np.ones_like(input_x)
    sample_len = input_shape[0]
    visit_len = input_shape[1]
    for i in range(visit_len):
        if input_x[:, i, :] == 0:
            mask[:, i, :] = 0
    mask = mask.reshape((-1, 1))
