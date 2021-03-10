from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from bayes_opt import BayesianOptimization


def c_index(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1

        if Time_survival[i] <= Time and Death[i] == 1:
            N_t[i, :] = 1

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


def imbalance_preprocess(train_features, time, labels):
    all_patient_time = []
    for i in range(time.shape[0]):
        one_patient_time = []
        for j in range(time.shape[1]):
            t = time[i, j, :].reshape(-1, 1)
            t_ = np.pad(t, ((0, train_features.shape[2] - time.shape[2]), (0, train_features.shape[3] - time.shape[2])),
                        'constant')
            one_patient_time.append(t_)
        all_patient_time.append(one_patient_time)
    features = np.concatenate((train_features, all_patient_time), axis=3)
    method = SMOTE()
    x_res, y_res = method.fit_sample(features.reshape(-1, features.shape[2] * features.shape[3]),
                                     labels.reshape(-1, labels.shape[2]))
    x_size = int(x_res.shape[0] / features.shape[1]) * features.shape[1]
    x_res = x_res[0:x_size, :]
    y_res = y_res[0:x_size, ]
    train_features_res = x_res.reshape(-1, features.shape[1], features.shape[2], features.shape[3])
    train_x_rex = train_features_res[:, :, :, 0:int(train_features_res.shape[3] / 2)]
    train_t_res = train_features_res[:, :, 0, int(train_features_res.shape[3] / 2)].reshape(-1, labels.shape[1],
                                                                                            labels.shape[2])
    train_y_res = y_res.reshape(-1, labels.shape[1], labels.shape[2])
    return train_x_rex, train_t_res, train_y_res


# split dataset
def split_data_set_gnn(dynamic_features, time, labels):
    time_steps = dynamic_features.shape[1]
    num_features = dynamic_features.shape[2]
    feature_dims = dynamic_features.shape[3]
    train_dynamic_features = {}
    train_labels = {}
    train_time = {}
    test_dynamic_features = {}
    test_time = {}
    test_labels = {}
    num = int(dynamic_features.shape[0] / 5)
    for i in range(4):
        test_dynamic_features[i] = dynamic_features[i * num:(i + 1) * num, :, :, :].reshape(-1, time_steps,
                                                                                            num_features, feature_dims)
        test_time[i] = time[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, 1)
        test_labels[i] = labels[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, 1)

    test_dynamic_features[4] = dynamic_features[4 * num:, :, :, :].reshape(-1, time_steps, num_features, feature_dims)
    test_time[4] = time[4 * num:, :, :].reshape(-1, time_steps, 1)
    test_labels[4] = labels[4 * num:, :, :].reshape(-1, time_steps, 1)

    train_dynamic_features[0] = dynamic_features[num:, :, :, :]
    train_time[0] = time[num:, :, :].reshape(-1, time_steps, 1)
    train_labels[0] = labels[num:, :, :]

    train_dynamic_features[1] = np.vstack((dynamic_features[0:num, :, :, :], dynamic_features[2 * num:, :, :, :]))
    train_time[1] = np.vstack((time[0:num, :, :], time[2 * num:, :, :])).reshape(-1, time_steps, 1)
    train_labels[1] = np.vstack((labels[0:num, :, :], labels[2 * num:, :, :]))

    train_dynamic_features[2] = np.vstack((dynamic_features[0:2 * num, :, :, :], dynamic_features[3 * num:, :, :, :]))
    train_time[2] = np.vstack((time[0:2 * num, :, :], time[3 * num:, :, :])).reshape(-1, time_steps, 1)
    train_labels[2] = np.vstack((labels[0:2 * num, :, :], labels[3 * num:, :, :]))

    train_dynamic_features[3] = np.vstack((dynamic_features[0:3 * num, :, :, :], dynamic_features[4 * num:, :, :, :]))
    train_time[3] = np.vstack((time[0:3 * num, :, :], time[4 * num:, :, :])).reshape(-1, time_steps, 1)
    train_labels[3] = np.vstack((labels[0:3 * num, :, :], labels[4 * num:, :, :]))

    train_dynamic_features[4] = dynamic_features[0:4 * num, :, :, :]
    train_time[4] = time[0:4 * num, :, :].reshape(-1, time_steps, 1)
    train_labels[4] = labels[0:4 * num, :, :]

    return train_dynamic_features, test_dynamic_features, train_time, test_time, train_labels, test_labels


# for loss_1:
# uncensored patient: the value is 1 when T is the event time
# censored patient: the va;ue is i in time horizon
def calculate_mask(time, label, num_category):
    time_step = time.shape[1]
    mask = np.zeros(shape=[0, time_step, num_category])
    batch = time.shape[0]
    for patient in range(batch):
        one_patient_mask = np.zeros(shape=[0, num_category])
        for visit in range(time_step):
            if label[patient, visit, 0] == 1.0:
                one_patient_visit_mask = np.zeros(shape=[num_category])
                patient_visit_t = int(time[patient, visit, 0])
                one_patient_visit_mask[patient_visit_t - 1] = 1.0
            else:
                one_patient_visit_mask = np.ones(shape=[num_category])
            one_patient_mask = np.concatenate((one_patient_mask, one_patient_visit_mask.reshape([-1, num_category])),
                                              axis=0)
        mask = np.concatenate((mask, one_patient_mask.reshape([-1, time_step, num_category])), axis=0)
    return mask


# loss_2:
# for uncensored patient: value is 1 from Time 0 to event time T
# for censored patient: value is 1 in time horizon (not used)
def calculate_mask_2(time, label, num_category):
    time_step = time.shape[1]
    mask = np.zeros(shape=[0, time_step, num_category])
    batch = time.shape[0]
    for patient in range(batch):
        one_patient_mask = np.zeros(shape=[0, num_category])
        for visit in range(time_step):
            if label[patient, visit, 0] == 1.0:
                one_patient_visit_mask = np.zeros(shape=[num_category])
                patient_visit_t = int(time[patient, visit, 0])
                one_patient_visit_mask[:patient_visit_t] = 1.0
            else:
                one_patient_visit_mask = np.ones(shape=[num_category])
            one_patient_mask = np.concatenate((one_patient_mask, one_patient_visit_mask.reshape([-1, num_category])),
                                              axis=0)
        mask = np.concatenate((mask, one_patient_mask.reshape([-1, time_step, num_category])), axis=0)
    return mask


def test():
    y = np.array([[0, 0, 1, 2, 3],
                  [1, 2, 3, 4, 5],
                  [1, 1, 1, 1, 1]])
    y = tf.convert_to_tensor(y)
    y = tf.cast(y, tf.float32)
    output = tf.keras.activations.softmax(y)
    print(output)


def test_test(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(name)

    print(path)
    print(os.path.dirname(__file__))
    print('------------------')


def modify_data():
    features = np.load('embedding_features_half_year_2dims.npy')
    time = np.load('pick_5_features_half_year.npy')
    for i in range(time.shape[0]):
        for j in range(time.shape[1]):
            if time[i, j, 0] == 0:
                time[i, j, 0] == 1.0
                print('修改第{}个病人第{}次记录完成'.format(i, j))
    np.save('pick_5_features_half_year_modify_time.npy', time)


def test_c_index(data, time, label, Time):
    data_array = np.zeros(shape=[0, 3])
    for i in data:
        data_array = np.concatenate((data_array, np.array(i).reshape(-1, 3)), axis=0)
    data = data_array
    time = np.array(time)
    label = np.array(label)
    n_all_all = []
    m_all_all = []
    for patient in range(data.shape[0]):  # 找到每个人的event time
        time__ = time[patient]
        time_ = time
        risk_ = data[:, int(time__ - 1)]
        label_ = label
        N, M = c_index_(risk_, time_, label_, Time)
        l_ = np.sum(M[patient, patient:])
        l_2 = np.sum(M[patient:, patient])

        print(np.sum(N[patient, patient:]), l_ + l_2 - M[patient, patient])
        n_all_all.append(N)
        m_all_all.append(M)
    print(n_all_all, m_all_all)


def c_index_(Prediction, Time_survival, Death):
    N = Prediction.shape[0]
    A = np.zeros(shape=[N, N])
    Q = np.zeros(shape=[N, N])
    N_t = np.zeros(shape=[N, N])
    for i in range(N):
        A[i, np.where(Time_survival[i] <= Time_survival)] = 1.0  # 存活时间比自己长的人
        Q[i, Prediction <= Prediction[i]] = 1.0  # 风险比自己低的人

        if Death[i] == 1:
            N_t[i] = 1

    return np.sum(Q * A * N_t), np.sum(A*N_t)

    # N = len(Prediction)
    # A = np.zeros((N, N))
    # Q = np.zeros((N, N))
    # N_t = np.zeros((N, N))
    # for i in range(N):
    #     A[i, np.where(Time_survival[i] <= Time_survival)] = 1
    #     Q[i, np.where(Prediction[i] >= Prediction)] = 1
    #
    #     if Time_survival[i] <= Time and Death[i] == 1:
    #         N_t[i, :] = 1
    #
    # Num = ((A) * N_t) * Q
    # Den = (A) * N_t

    # return Num, Den


def calculate_c_index(prediction, time_survival, death, Time):
    time_step = prediction.shape[1]
    batch = prediction.shape[0]
    time_survival = time_survival.reshape([-1, time_step, 1])
    death = death.reshape([-1, time_step, 1])
    c_index = []
    for time_index in range(time_step):
        n_one_visit = []
        m_one_viist = []
        survival_time = time_survival[:, time_index, 0]
        death_ = death[:, time_index, :]
        for patient in range(batch):
            time_patient = int(time_survival[patient, time_index, 0]) - 1
            risk = prediction[:, time_index, time_patient]
            n_one_patient, m_one_patient = c_index_(risk, survival_time, death_)

            n_one_visit.append(n_one_patient)
            m_one_viist.append(m_one_patient)

        if np.sum(m_one_viist) == 0.0 and np.sum(n_one_visit) == 0.0:
            c_index.append(-1)
        else:
            c_index.append(np.sum(n_one_visit) / np.sum(m_one_viist))
        return np.mean(c_index)


if __name__ == "__main__":
    score = [[0.1, 0.5, 0.0],
             [0.5, 0.1, 0.2],
             [0.0, 0.0, 0.1]]

    time_survival = [2, 1, 3]
    death = [1, 1, 0]

    data_array = np.zeros(shape=[0, 3])
    for i in score:
        data_array = np.concatenate((data_array, np.array(i).reshape(-1, 3)), axis=0)
    data = data_array.reshape(3, -1, 3)
    time = np.array(time_survival).reshape(-1, 3, 1)
    label = np.array(death).reshape(-1, 3, 1)
    calculate_c_index(data, time, label, 10)
