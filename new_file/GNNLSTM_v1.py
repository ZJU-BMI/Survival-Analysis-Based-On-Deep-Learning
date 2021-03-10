import tensorflow as tf
from tensorflow.python.keras.models import Model
import os
from data import DataSet
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
import numpy as np
from lifelines.utils import concordance_index
from new_file.utils import *
tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def calculate_batch(data):
    return data.shape[0]


def get_time(time, patient_index, time_index, index):
    sess = tf.compat.v1.Session()
    time_ = time[patient_index, time_index, index].eval(session=sess)
    return time_


class GNNLSTMModel(Model):
    def __init__(self, time_step, hidden_size, num_category, num_features, feature_dims):
        super().__init__(name='GNNLSTMModel')
        self.time_step = time_step
        self.num_category = num_category
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.hidden_size = hidden_size
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size*num_features)

    def build(self, input_shape):
        shape_attention_weight = tf.TensorShape((self.num_features, self.feature_dims))
        shape_graph_init_weight = tf.TensorShape((self.feature_dims*2, 1))
        graph_init_weight = tf.keras.initializers.TruncatedNormal(stddev=1.0)
        aggregation_weight_out = tf.keras.initializers.TruncatedNormal(stddev=1.0)
        aggregation_weight_in = tf.keras.initializers.TruncatedNormal(stddev=1.0)
        output_weight = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        graph_init_bias = tf.keras.initializers.RandomNormal([0.0])

        self.attention_weight = self.add_weight(name='attention_weight',
                                                shape=shape_attention_weight,
                                                initializer='truncated_normal',
                                                trainable=True)

        self.graph_iniit_weight = tf.Variable(initial_value=graph_init_weight(shape=shape_graph_init_weight),
                                              trainable=True,
                                              name='graph_init_weight')

        self.graph_init_bias = tf.Variable(initial_value=graph_init_bias(shape=[1,]),
                                           trainable=True,
                                           name='graph_init_bias')

        self.aggregation_weight_out = tf.Variable(initial_value=aggregation_weight_out(shape=[self.feature_dims, self.feature_dims]),
                                                  trainable=True,
                                                  name='weight_out')

        self.aggregation_weight_in = tf.Variable(initial_value=aggregation_weight_in(shape=[self.feature_dims, self.feature_dims]),
                                                 trainable=True,
                                                 name='weight_in')

        self.aggregation_bias_out = self.add_weight(name='bias_out',
                                                    shape=[self.num_features, 1],
                                                    initializer='truncated_normal',
                                                    trainable=True)

        self.aggregation_bias_in = self.add_weight(name='bias_oin',
                                                   shape=[self.num_features, 1],
                                                   initializer='random_normal',
                                                   trainable=True)

        self.output_weight = tf.Variable(initial_value=output_weight(shape=[self.num_features*self.hidden_size*2, self.num_category]),
                                         trainable=True,
                                         name='output_weight')

        self.output_bias = self.add_weight(name='output_bias',
                                           shape=[self.num_category],
                                           initializer='random_normal',
                                           trainable=True)

        super(GNNLSTMModel, self).build(input_shape)

    def call(self, input_data):
        features, time, label, mask_, mask_2 = input_data
        x = features
        attention_output = self.attention_mechanism(features)
        features += attention_output
        states_fw, states_bw = self.state_update(features, self.hidden_size)
        states = []
        for i in range(self.time_step):
            state = tf.concat((states_fw[i], states_bw[i]), axis=1)
            states.append(state)
        states = tf.transpose(states, [1, 0, 2])  # [patient_num, time_step, hidden_size*num_features*2]
        output = self.output(states)
        mask, length = self.length(x)
        mask = tf.reshape(mask, [-1, self.time_step, 1])
        mask = tf.tile(mask, [1, 1, self.num_category])
        prediction = tf.keras.activations.softmax(output, 2)
        prediction = tf.multiply(mask, prediction)

        # loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=prediction))

        loss_1 = self.loss_1(prediction, label, mask_)  # 对于死亡的病人来说，仅仅用了死亡当天的结果
        # loss_2 = self.neg_likelihood_loss(prediction, time, label, batch)

        loss_3 = self.loss_2(prediction, label, mask_2)  # 对于死亡的病人来说，用了从第一天到最后一天的结果
        return prediction, loss_1, 0, loss_3

    def attention_mechanism(self, x):
        weight = tf.keras.activations.softmax(self.attention_weight, axis=1)
        weight = tf.tile(tf.reshape(weight, [-1, 1, self.num_features, self.feature_dims]), [tf.shape(x)[0], self.time_step, 1, 1])
        attention_output = tf.multiply(x, weight)
        return attention_output

    def state_update(self, x, hidden_size):
        global new_h, new_c
        states_fw = []
        states_bw = []
        graph = self.init_graph(x)
        state_aggregation_list = []
        for i in range(self.time_step):
            graph_i = graph[:, i, :, :]
            x_i = x[:, i, :, :]
            state_aggregation = self.state_aggregation(x_i, graph_i)
            state_aggregation_list.append(state_aggregation)

        for i in range(self.time_step):
            x_i = x[:, i, :, :]
            state_aggregation = state_aggregation_list[i]
            c = tf.reshape(state_aggregation, [-1, hidden_size*self.num_features])
            input_x = tf.reshape(x_i, [-1, self.num_features*self.feature_dims])
            if i == 0:
                h = tf.zeros_like(c)
            else:
                h = new_h
                c = new_c
            states = (c, h)
            new_h, (new_c, _) = self.LSTM_Cell(input_x, states)
            states_fw.append(new_h)

        for j in range(self.time_step):
            if j == 0:
                x_i = tf.reshape(x[:, j, :, :], [-1, self.num_features*self.feature_dims])
                state_aggregation = state_aggregation_list[j]
                c = tf.reshape(state_aggregation, [-1, hidden_size*self.num_features])
                h = tf.zeros_like(c)
                states = (c, h)
                new_h, (new_c, _) = self.LSTM_Cell(x_i, states)
                states_bw.append(new_h)
            else:
                for i in range(j+1):
                    l = j-i  # inverse order
                    x_i = tf.reshape(x[:, l, :, :], [-1, self.num_features*self.feature_dims])
                    state_aggregation = state_aggregation_list[l]
                    c = tf.reshape(state_aggregation, [-1, hidden_size*self.num_features])
                    if l == j:
                        h = tf.zeros_like(c)
                    else:
                        h = new_h
                        c = new_c
                    states = (c, h)
                    new_h, (new_c, _) = self.LSTM_Cell(x_i, states)
                states_bw.append(new_h)
        return states_fw, states_bw

    # representation each feature in graph using neighboring features
    def state_aggregation(self, x, graph):
        batch = tf.shape(x)[0]
        x_ = tf.reshape(x, [batch, self.num_features, self.feature_dims])
        graph_ = tf.reshape(graph, [batch, self.num_features, self.num_features])
        w_out_all = tf.tile(tf.reshape(self.aggregation_weight_out, [-1, self.feature_dims, self.feature_dims]), [batch, 1, 1])
        w_out_b_all = tf.tile(tf.reshape(self.aggregation_bias_out, [-1, self.num_features, 1]), [batch, 1, self.feature_dims])
        x_ = tf.matmul(x_, w_out_all) + w_out_b_all
        states = tf.matmul(graph_, x_)

        w_in_all = tf.tile(tf.reshape(self.aggregation_weight_in, [-1, self.feature_dims, self.feature_dims]), [batch, 1, 1])
        w_in_b_all = tf.tile(tf.reshape(self.aggregation_bias_in, [-1, self.num_features, 1]), [batch, 1, self.feature_dims])

        state_aggregation = tf.matmul(states, w_in_all) + w_in_b_all
        return state_aggregation

    # 初始化图embedding
    def init_graph(self, x):
        features = tf.reshape(x, [-1, self.num_features, self.feature_dims])
        batch = tf.shape(features)[0]  # patient_num * time_steps
        a = tf.tile(features, [1, self.num_features, 1])
        b = tf.tile(features, [1, 1, self.num_features])
        a = tf.reshape(a, [batch, self.num_features, self.num_features, self.feature_dims])
        b = tf.reshape(b, [batch, self.num_features, self.num_features, self.feature_dims])
        m = tf.concat((a, b), axis=3)
        m = tf.reshape(m, [batch, self.num_features*self.num_features, 2*self.feature_dims])
        w_w = tf.tile(self.graph_iniit_weight, [batch, 1])
        w_w = tf.reshape(w_w, [batch, 2*self.feature_dims, 1])
        w_b = tf.tile(self.graph_init_bias, [self.num_features*batch])
        w_b = tf.reshape(w_b, [batch, self.num_features, 1])
        weight = tf.reshape(tf.matmul(m, w_w), [batch, self.num_features, self.num_features]) + w_b
        weight = tf.linalg.band_part(weight, 0, -1)
        weight_trans = tf.transpose(weight, [0, 2, 1])
        weight_ = weight + weight_trans
        adjacency = tf.ones([self.num_features, self.num_features]) - tf.eye(self.num_features, num_columns=self.num_features)
        adjacency_all = tf.reshape(tf.tile(adjacency, [batch, 1]), [batch, self.num_features, self.num_features])
        graph_ = tf.multiply(weight_, adjacency_all)
        graph = tf.reshape(tf.keras.activations.softmax(graph_, axis=1), [-1, self.time_step, self.num_features, self.num_features])
        return graph

    def output(self, states):
        all_states = tf.reshape(states, [-1, self.time_step, self.num_features * self.hidden_size * 2])
        output = tf.zeros(shape=[tf.shape(all_states)[0], 0, self.num_category])
        for i in range(self.time_step):
            output_ = tf.matmul(all_states[:, i, :], self.output_weight) + self.output_bias
            # output_ = tf.keras.activations.softmax(output_)
            output_ = tf.keras.activations.softmax(output_)
            output = tf.concat((output, tf.reshape(output_, [-1, 1, self.num_category])), axis=1)

        return output

    def length(self, x):
        m = tf.reduce_max(tf.abs(x), 3)
        mask = tf.sign(tf.reduce_max(tf.abs(m), 2))
        length = tf.reduce_sum(mask, 1)
        return mask, length

    def loss_1(self, prediction, label, mask):
        prediction = tf.multiply(mask, prediction)
        # for uncensored patients(死亡的patient)
        tmp_1 = tf.reduce_sum(prediction, 2)
        tmp_1 = tf.reshape(tmp_1, [-1, self.time_step, 1])
        tmp_1 = label * tmp_1
        # for censored patient(没有死亡的patient
        tmp_2 = tf.reduce_sum(prediction[:, :, :self.num_category], 2)
        tmp_2 = tf.reshape(tmp_2, [-1, self.time_step, 1])
        tmp_2 = (1 - label) * tmp_2
        return - tf.reduce_mean(tmp_1 + tmp_2)

    def loss_2(self, prediction, label, mask):
        prediction = tf.multiply(mask, prediction)
        # for uncensored patients
        tmp_1 = tf.reduce_sum(prediction, 2)
        tmp_1 = tf.reshape(tmp_1, [-1, self.time_step, 1])
        tmp_1 = label * tmp_1
        return - tf.reduce_mean(tmp_1)

    # def neg_likelihood_loss(self, prediction, time, label, batch):
    #     neg_likelihood_all = 0.0
    #     for time_index in range(self.time_step):
    #         neg_likelihood_ = 0.0
    #         for patient in range(batch):
    #             time_patient_ = get_time(time, patient, time_index, 0)
    #             prediction_ = prediction[:, time_index, int(time_patient_)-1]  # [patient_num, ]
    #             time_ = time[:, time_index, :]
    #             y_ = label[: time_index, :]
    #
    #             y_ = tf.reshape(y_, [-1])  # event
    #             time_ = tf.reshape(time_, [-1,])  # survival time
    #             risk = tf.reshape(prediction_, [-1,])  # hazard risk
    #
    #             sort_idx = tf.argsort(time_, direction='DESCENDING')
    #             y_ = tf.gather(y_, sort_idx)
    #             risk = tf.gather(risk, sort_idx)
    #
    #             hazard_ratio = tf.exp(risk)
    #             log_risk = tf.math.log(tf.cumsum(hazard_ratio))
    #             uncensored_likelihood = risk - log_risk
    #             uncensored_likelihood = tf.multiply(uncensored_likelihood, y_)
    #             neg_likelihood_ += - tf.reduce_sum(uncensored_likelihood)
    #         neg_likelihood_all += neg_likelihood_
    #
    #     return neg_likelihood_all


def neg_likelihood_loss(prediction, time, label):
    neg_likelihood_all = 0.0
    time_step = prediction.shape[1]
    batch = prediction.shape[0]
    for time_index in range(time_step):
        neg_likelihood_ = 0.0
        for patient in range(batch):
            time_patient_ = time[patient, time_index, 0]
            prediction_ = prediction[:, time_index, int(time_patient_)-1]  # [patient_num, ]
            time_ = time[:, time_index, :]
            y_ = label[:, time_index, :]

            y_ = tf.reshape(y_, [-1])  # event
            y_ = tf.cast(y_, tf.float32)
            time_ = tf.reshape(time_, [-1,])  # survival time
            risk = tf.reshape(prediction_, [-1,])  # hazard risk

            sort_idx = tf.argsort(time_, direction='DESCENDING')
            y_ = tf.gather(y_, sort_idx)
            risk = tf.gather(risk, sort_idx)

            hazard_ratio = tf.exp(risk)
            log_risk = tf.math.log(tf.cumsum(hazard_ratio))
            uncensored_likelihood = risk - log_risk
            uncensored_likelihood = tf.multiply(uncensored_likelihood, y_)
            neg_likelihood_ += - tf.reduce_sum(uncensored_likelihood)
        neg_likelihood_all += neg_likelihood_

    return neg_likelihood_all


# train model
def train(learning_rate, imbalance_1, imbalance_2, l2_regularization):
    learning_rate = 10 ** int(learning_rate)
    imbalance_1 = 10 ** int(imbalance_1)
    imbalance_2 = 10 ** int(imbalance_2)
    l2_regularization = 10 ** int(l2_regularization)
    # learning_rate=0.1
    # imbalance_1= 0.01
    # imbalance_2=0.01
    # l2_regularization=1e-5
    print('learning_rate={} imbalance_1={} imbalance_1={} l2_regularization={}'.format(learning_rate, imbalance_1, imbalance_2, l2_regularization))

    features = np.load('embedding_features_half_year_2dims.npy')
    labels = np.load('pick_5_labels_half_year.npy')
    time = np.load('pick_5_features_half_year_modify_time.npy')[:, :, 0]
    time = time.reshape(-1, time.shape[1], 1)

    time_steps = time.shape[1]
    num_category = int(np.max(time))+1
    num_features = features.shape[2]
    feature_dims = features.shape[3]

    epochs = 15
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    train_features, test_features, train_time, test_time, train_labels, test_labels = split_data_set_gnn(features, time, labels)
    all_c_index = []
    for i in range(5):
        # train_x_res, train_t_res, train_labels_res = imbalance_preprocess(train_features[i], train_time[i], train_labels[i])
        train_set = DataSet(train_features[i], train_time[i], train_labels[i])
        model = GNNLSTMModel(time_step=time_steps, hidden_size=feature_dims,
                             num_category=num_category, num_features=num_features, feature_dims=feature_dims)

        test_c_index = do_experiment(validate_epoch=i, epochs=epochs, train_set=train_set, imbalance_1=imbalance_1,
                                     imbalance_2=imbalance_2,num_category=num_category,
                                     model=model, test_t=test_time[i], test_y=test_labels[i],
                                     test_features=test_features[i], optimizer=optimizer, time_steps=time_steps, l2_regularization=l2_regularization)
        all_c_index.append(test_c_index)
    return np.mean(all_c_index)


def do_experiment(validate_epoch, epochs, train_set, imbalance_1, imbalance_2, num_category, model, time_steps, test_t, test_y, test_features, optimizer, l2_regularization):
    logged = set()
    max_test = -99
    batch = 64
    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            train_x, train_t, train_y = train_set.next_batch(batch_size=batch)
            mask = calculate_mask(train_t, train_y, num_category)
            mask_2 = calculate_mask_2(train_t, train_y, num_category)
            prediction, loss_1, loss_2, loss_3 = model([train_x, train_t, train_y, mask, mask_2])
            loss_2 = neg_likelihood_loss(prediction, train_t, train_y)
            loss = loss_1 + loss_2 * imbalance_1 + loss_3 * imbalance_2
            # risk in the horizon time
            # risk = prediction[:, :, -2]
            # c_index_train = 0.0
            # for time_index in range(time_steps):
            #     risk_ = risk[:, time_index]
            #     time_survival_ = train_t[:, time_index, :].reshape([-1, ])
            #     death_ = train_y[:, time_index, :].reshape([-1, ])
            #     time_ = num_category
            #     c_index_ = c_index(risk_, time_survival_, death_, time_)
            #     # c_index_test_ = concordance_index(time_survival_, risk_, death_)
            #     c_index_train += c_index_
            # c_index_train = c_index_train / time_steps
            c_index_train = calculate_c_index(prediction=prediction, time_survival=train_t, death=train_y, Time=num_category)
            for weight in model.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                mask_test = calculate_mask(test_t, test_y, num_category)
                mask_test_2 = calculate_mask_2(test_t, test_y, num_category)
                prediction_test, loss_1_test, loss_2_test, loss_3_test = model(
                    [test_features, test_t, test_y, mask_test, mask_test_2])

                tmp_test = np.sum(prediction_test[:, :, :-1], 2)  # endpoint prediction(risk)
                test_label_ = test_y.reshape([-1, ])
                test_prediction_label = tmp_test.reshape([-1, ])
                auc = roc_auc_score(test_label_, test_prediction_label)
                fpr, tpr, thresholds = roc_curve(test_label_, test_prediction_label, pos_label=1)
                threshold = thresholds[np.argmax(tpr - fpr)]
                y_pred_label = (test_prediction_label >= threshold) * 1
                acc = accuracy_score(test_label_, y_pred_label)
                recall = recall_score(test_label_, y_pred_label)
                precision = precision_score(test_label_, y_pred_label)
                f1 = f1_score(test_label_, y_pred_label)

                # risk = prediction_test[:, :, -2]
                # c_index_test = 0.0
                # for time_index in range(time_steps):
                #     risk_ = risk[:, time_index]
                #     time_survival_ = test_t[:, time_index, :].reshape([-1, ])
                #     death_ = test_y[:, time_index, :].reshape([-1, ])
                #     time_ = num_category
                #     c_index_test_ = c_index(risk_, time_survival_, death_, time_)
                #     # c_index_test_ = concordance_index(time_survival_, risk_, death_)
                #     c_index_test += c_index_test_
                # c_index_test = c_index_test/time_steps
                c_index_test = calculate_c_index(prediction=prediction_test, time_survival=test_t, death=test_y, Time=num_category)

                if c_index_test > max_test:
                    stop_flag = 0.0
                    max_test = c_index_test
                else:
                    stop_flag += 1

                if stop_flag > 5:
                    break

                print('validate_epoch--{}--epoch---{}---loss_1_train__{}---loss_2_train__{}---loss_gen--{}----'
                      'c_index_train__{}'
                      '---c_index_test---{}--test_auc---{}---'
                      'test_acc---{}---test_f1---{}---test_recall___{}---precision___{}---count--{}'
                      .format(validate_epoch,
                              train_set.epoch_completed,
                              loss_1,
                              loss_2,
                              loss_3,
                              c_index_train,
                              c_index_test,
                              auc,
                              acc,
                              f1,
                              recall,
                              precision,
                              stop_flag))

    return c_index_test


if __name__ == "__main__":
    train(learning_rate=-1, imbalance_1=-3, imbalance_2=-3, l2_regularization=-5)
    # test_test('12_9_survival_analysis_train.txt')
    # mse_all = []
    # mae_all = []
    # environment_train = BayesianOptimization(
    #     train, {
    #         'learning_rate': (-6, 1),
    #         'imbalance_1': (-6, 0),
    #         'imbalance_2': (-6, 0),
    #         'l2_regularization': (-6, -1),
    #     }
    # )
    # environment_train.maximize()
    # print(environment_train.max)

