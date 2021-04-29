import numpy as np
from utils import *
from tensorflow.keras import *
from tensorflow import keras
from test.layer.Decoder import *
from test.layer.Encoder import Encoder
from test.S2S import S2S
from test.RNN import RNN
from test.output_layer import *
from utils import *
from process.Process import SAVE_DIR
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
import copy

batch_size = 256
epochs = 220
MAX_CINDEX = 0
hidden_size1_final = 0
hidden_size2_final = 0
hidden_size3_final = 0
hidden_size4_final = 0
a1_final = 0
a2_final = 0
a3_final = 0
learning_rate_final = 0
l2_regularization_final = 0
MASK_RATE = 1
SHUFFLE_RATE = 1
def read_mimic_data():
    train_data = np.load('data/mimic_train.npy')
    test_data = np.load('data/mimic_test.npy')
    train_features = train_data[:, :, 2:64].reshape(-1, 3, 62)
    train_label = train_data[:, :, 0].reshape(-1, 3)
    train_time = train_data[:, :, 1].reshape(-1, 3)
    test_features = test_data[:, :, 2:64].reshape(-1, 3, 62)
    test_label = test_data[:, :, 0].reshape(-1, 3)
    test_time = test_data[:, :, 1].reshape(-1, 3)
    return DataSet(train_features, train_time, train_label), DataSet(test_features, test_time, test_label)
def read_mimic_data_with_time_interval():
    train_data = np.load('data/mimic_train2.npy')
    test_data = np.load('data/mimic_test2.npy')
    train_features = train_data[:, :, 2:64].reshape(-1, 3, 62)
    train_label = train_data[:, :, 0].reshape(-1, 3)
    train_time = train_data[:, :, 1].reshape(-1, 3)
    train_day = train_data[:, :, 64].reshape(-1, 3, 1)
    test_features = test_data[:, :, 2:64].reshape(-1, 3, 62)
    test_label = test_data[:, :, 0].reshape(-1, 3)
    test_time = test_data[:, :, 1].reshape(-1, 3)
    test_day = test_data[:, :, 64].reshape(-1, 3, 1)
    return DataSet2(train_features, train_time, train_label, train_day), DataSet2(test_features, test_time, test_label, test_day)



def train_model_mimic_all(hidden_size, learning_rate, l2_regularization, time_windows):


    target_label = time_windows
    train_set, test_set = read_mimic_data()
    feature = train_set.x
    feature_dims = feature.shape[2]
    visit_len = feature.shape[1]

    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    # shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)

    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))
    # discriminator = DISCRIMINATOR(hidden_size=hidden_size, feature_dims=28)
    encoder = Encoder(hidden_size=32, model_type='LSTM')
    decoder = Decoder(hidden_size=32, feature_dims=feature_dims, model_type='LSTM')
    sap = SAP(hidden_size=256)
    mlp = MLP2(hidden_size=64)
    logged = set()
    max_loss = 0.01
    max_pace = 0.0001
    result = []
    shuffle_index = [2,1,0]
    count = 0
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_, input_t_train_, input_y_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 做mask
            # mask_index = int(np.random.random() * visit_len)
            mask_index = int(np.random.random() * (visit_len-1))

            if MASK_RATE==0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:

                # mask_index = 2
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select = np.random.random()
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            # 生成预测的序列
            trajectory_encode_last_h, trajectory_encode_h_list = encoder(mask_input_x_train, batch=batch_size)
            # context_state = mlp(trajectory_encode_last_h)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder(trajectory_encode_last_h,
                                                                                  predicted_visit=visit_len,
                                                                                  batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))
            mask_input_x_train_add = copy.deepcopy(mask_input_x_train)
            mask_input_x_train_add[:, mask_index, :] = predicted_trajectory_x_train[:, mask_index, :]
            mask_input_x_train_trajectory_generation_decode_h,  mask_input_x_train_trajectory_generation_h_list = encoder(mask_input_x_train_add,
                                                                                    batch=batch_size)

            real_decode_h, real_trajectory_encode_h_list = encoder(input_x_train_, batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            for v in range(visit_len):
                predicted_output = sap(real_trajectory_encode_h_list[:, v, :])
                label = input_y_train_[:, v]
                label = label.reshape((-1, 1)).astype('float32')
                clf_loss += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predicted_output))
                neg_likelihood_loss += partial_log_likelihood(predicted_output, input_t_train_[:, v], label)
            survival_prediction_loss = tf.add(clf_loss, neg_likelihood_loss)

            if SHUFFLE_RATE==0:
                shuffled_input_x_train=input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]

                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1

            # shuffled_input_x_train = input_x_train_[:, shuffle_index, :]

            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder(shuffled_input_x_train,
                                                                                    batch=batch_size)


            # 对比学习
            contrast_loss_matrix = tf.matmul(mlp(shuffled_generated_decode_h), tf.transpose(mlp(real_decode_h)))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1)

            contrast_loss_matrix2 = tf.matmul(mlp(real_decode_h), tf.transpose(mlp(shuffled_generated_decode_h)))
            contrast_loss_numerator2 = tf.linalg.diag_part(contrast_loss_matrix2)
            contrast_loss_denominator2 = tf.reduce_sum(tf.math.exp(contrast_loss_matrix2),
                                                      axis=1)

            contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))-tf.reduce_mean(contrast_loss_numerator2 - tf.math.log(contrast_loss_denominator2))

            # 对比学习2
            contrast_loss_trajectory_generation = tf.matmul(mask_input_x_train_trajectory_generation_decode_h, tf.transpose(real_decode_h))
            contrast_loss_trajectory_generation_numerator = tf.linalg.diag_part(contrast_loss_trajectory_generation)
            contrast_loss_trajectory_generation_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_trajectory_generation),
                                                      axis=1)

            contrast_loss_trajectory_generation2 = tf.matmul(real_decode_h, tf.transpose(mask_input_x_train_trajectory_generation_decode_h))
            contrast_loss_trajectory_generation_numerator2 = tf.linalg.diag_part(contrast_loss_trajectory_generation2)
            contrast_loss_trajectory_generation_denominator2 = tf.reduce_sum(tf.math.exp(contrast_loss_trajectory_generation2),
                                                       axis=1)

            contrast_loss = -tf.reduce_mean(
                contrast_loss_numerator - tf.math.log(contrast_loss_denominator)) - tf.reduce_mean(
                contrast_loss_numerator2 - tf.math.log(contrast_loss_denominator2))
            contrast_loss2 = -tf.reduce_mean(
                contrast_loss_trajectory_generation_numerator - tf.math.log(contrast_loss_trajectory_generation_denominator)) - tf.reduce_mean(
                contrast_loss_trajectory_generation_numerator2 - tf.math.log(contrast_loss_trajectory_generation_denominator2))
            whole_loss = clf_loss + gen_mse_loss * 0.3 + neg_likelihood_loss * 1 + contrast_loss * 0.01 + contrast_loss2*0.1
            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            sap_variables = [var for var in sap.trainable_variables]
            for weight in sap.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            mlp_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            variables = sap_variables + encoder_variables + decoder_variables + mlp_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            # if train_set.epoch_completed == 1:
            #     s2s.load_weights('S2S_weight_{}_v1.h5'.format(target_label))


            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder(input_x_test, batch=batch_test)
                predicted_output_test_list = np.zeros((batch_test, 0))
                for v in range(visit_len):
                    predicted_output_test = sap(real_trajectory_encode_h_list_test[:, v, :]).numpy()
                    predicted_output_test_list = np.concatenate([predicted_output_test_list, predicted_output_test],
                                                                axis=1)

                label_test = input_y_test.reshape((-1, 1)).astype('float32')
                predicted_output_test_list = predicted_output_test_list.reshape((-1, 1))
                try:
                    y_pred_label, auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(
                        label_test, predicted_output_test_list)
                    c_index = concordance_index(input_t_test.reshape((-1, 1)), -predicted_output_test_list,
                                                label_test)
                    result.append([c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test])

                except:
                    break

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print(
                        '----epoch:{}, whole_loss:{}, contrast_loss:{},contrast_loss2:{},clf_loss:{},gen_loss:{}, c_index:{},auc:{},precision_test:{},recall_test:{},f_score_test:{},accuracy_test:{}, '
                        '-count:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss, contrast_loss2, clf_loss+neg_likelihood_loss,
                                           gen_mse_loss, c_index,
                                           auc_test, precision_test, recall_test, f_score_test, accuracy_test, count))



        tf.compat.v1.reset_default_graph()
    result = np.array(result)
    max_i = np.where(result[:, 0] == result[:, 0].max())
    print(np.squeeze(result[max_i, :]))
    return np.squeeze(result[max_i, :])


def train_model_mimic_timelstm1(hidden_size, learning_rate, l2_regularization, time_windows):
    """
    timelstm1

    :param hidden_size:hidden_size
    :param learning_rate:learning_rate
    :param l2_regularization:l2_regularization
    :param time_windows:prediction survival time windows
    :return:[c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test]
    """

    target_label = time_windows
    train_set, test_set = read_mimic_data_with_time_interval()
    feature = train_set.x
    feature_dims = feature.shape[2]
    visit_len = feature.shape[1]

    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    # shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)

    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))
    # discriminator = DISCRIMINATOR(hidden_size=hidden_size, feature_dims=28)
    encoder = Encoder(hidden_size=32, model_type='LSTM')
    decoder = Decoder(hidden_size=32, feature_dims=feature_dims, model_type='TimeLSTM1')
    sap = SAP(hidden_size=256)
    mlp = MLP2(hidden_size=64)
    logged = set()
    max_loss = 0.01
    max_pace = 0.0001
    result = []
    shuffle_index = [2,1,0]
    count = 0
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_, input_t_train_, input_y_train_, input_day_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 做mask
            # mask_index = int(np.random.random() * visit_len)
            mask_index = int(np.random.random() * (visit_len-1))

            if MASK_RATE==0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:

                # mask_index = 2
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select = np.random.random()
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            # 生成预测的序列
            trajectory_encode_last_h, trajectory_encode_h_list = encoder((mask_input_x_train), batch=batch_size)
            # context_state = mlp(trajectory_encode_last_h)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder((trajectory_encode_last_h,input_day_train_),
                                                                                  predicted_visit=visit_len,
                                                                                  batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))

            real_decode_h, real_trajectory_encode_h_list = encoder((input_x_train_), batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            for v in range(visit_len):
                predicted_output = sap(real_trajectory_encode_h_list[:, v, :])
                label = input_y_train_[:, v]
                label = label.reshape((-1, 1)).astype('float32')
                clf_loss += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predicted_output))
                neg_likelihood_loss += partial_log_likelihood(predicted_output, input_t_train_[:, v], label)
            survival_prediction_loss = tf.add(clf_loss, neg_likelihood_loss)

            if SHUFFLE_RATE==0:
                shuffled_input_x_train=input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]

                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1

            # shuffled_input_x_train = input_x_train_[:, shuffle_index, :]

            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder((shuffled_input_x_train),
                                                                                    batch=batch_size)

            # contrast_loss_matrix = tf.matmul(mlp(s_h), tf.transpose(mlp(r_h)))
            # contrast_loss_matrix = -tf.keras.losses.cosine_similarity(tf.expand_dims(mlp(s_h), 1),
            #                                                           tf.expand_dims(mlp(r_h), 0))
            # 对比学习
            contrast_loss_matrix = tf.matmul(mlp(shuffled_generated_decode_h), tf.transpose(mlp(real_decode_h)))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1) - contrast_loss_numerator

            contrast_loss_matrix2 = tf.matmul(mlp(real_decode_h), tf.transpose(mlp(shuffled_generated_decode_h)))
            contrast_loss_numerator2 = tf.linalg.diag_part(contrast_loss_matrix2)
            contrast_loss_denominator2 = tf.reduce_sum(tf.math.exp(contrast_loss_matrix2),
                                                      axis=1) - contrast_loss_numerator2

            contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))-tf.reduce_mean(contrast_loss_numerator2 - tf.math.log(contrast_loss_denominator2))
            whole_loss = clf_loss + gen_mse_loss * 0.3 + neg_likelihood_loss * 1 + contrast_loss * 0.01
            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            sap_variables = [var for var in sap.trainable_variables]
            for weight in sap.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            mlp_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            variables = sap_variables + encoder_variables + decoder_variables + mlp_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            # if train_set.epoch_completed == 1:
            #     s2s.load_weights('S2S_weight_{}_v1.h5'.format(target_label))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                input_day_test = test_set.day
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder((input_x_test), batch=batch_test)
                predicted_output_test_list = np.zeros((batch_test, 0))
                for v in range(visit_len):
                    predicted_output_test = sap(real_trajectory_encode_h_list_test[:, v, :]).numpy()
                    predicted_output_test_list = np.concatenate([predicted_output_test_list, predicted_output_test],
                                                                axis=1)

                label_test = input_y_test.reshape((-1, 1)).astype('float32')
                predicted_output_test_list = predicted_output_test_list.reshape((-1, 1))
                try:
                    y_pred_label, auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(
                        label_test, predicted_output_test_list)
                    c_index = concordance_index(input_t_test.reshape((-1, 1)), -predicted_output_test_list,
                                                label_test)
                    result.append([c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test])

                except:
                    break

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print(
                        '----epoch:{}, whole_loss:{}, contrast_loss:{},survaval_loss:{},gen_loss:{}, c_index:{},auc:{},precision_test:{},recall_test:{},f_score_test:{},accuracy_test:{}, '
                        '-count:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss, clf_loss+neg_likelihood_loss,
                                           gen_mse_loss, c_index,
                                           auc_test, precision_test, recall_test, f_score_test, accuracy_test, count))

        # if train_set.epoch_completed == epochs - 1:
        #     encoder.save_weights('save/train_model_test_encoder_weight_{}_mimic.h5'.format(target_label))
        #     decoder.save_weights('save/train_model_test_decoder_weight_{}_mimic.h5'.format(target_label))
        #     sap.save_weights('save/train_model_test_sap_weight_{}_mimic.h5'.format(target_label))

        tf.compat.v1.reset_default_graph()
    result = np.array(result)
    max_i = np.where(result[:, 0] == result[:, 0].max())
    print(np.squeeze(result[max_i, :]))
    return np.squeeze(result[max_i, :])
def train_model_mimic_timelstm2(hidden_size, learning_rate, l2_regularization, time_windows):
    """
    timelstm1

    :param hidden_size:hidden_size
    :param learning_rate:learning_rate
    :param l2_regularization:l2_regularization
    :param time_windows:prediction survival time windows
    :return:[c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test]
    """

    target_label = time_windows
    train_set, test_set = read_mimic_data_with_time_interval()
    feature = train_set.x
    feature_dims = feature.shape[2]
    visit_len = feature.shape[1]

    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    # shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)

    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))
    # discriminator = DISCRIMINATOR(hidden_size=hidden_size, feature_dims=28)
    encoder = Encoder(hidden_size=32, model_type='LSTM')
    decoder = Decoder(hidden_size=32, feature_dims=feature_dims, model_type='TimeLSTM2')
    sap = SAP(hidden_size=256)
    mlp = MLP2(hidden_size=64)
    logged = set()
    max_loss = 0.01
    max_pace = 0.0001
    result = []
    shuffle_index = [2,1,0]
    count = 0
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_, input_t_train_, input_y_train_, input_day_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 做mask
            # mask_index = int(np.random.random() * visit_len)
            mask_index = int(np.random.random() * (visit_len-1))

            if MASK_RATE==0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:

                # mask_index = 2
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select = np.random.random()
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            # 生成预测的序列
            trajectory_encode_last_h, trajectory_encode_h_list = encoder((mask_input_x_train), batch=batch_size)
            # context_state = mlp(trajectory_encode_last_h)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder((trajectory_encode_last_h,input_day_train_),
                                                                                  predicted_visit=visit_len,
                                                                                  batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))

            real_decode_h, real_trajectory_encode_h_list = encoder((input_x_train_), batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            for v in range(visit_len):
                predicted_output = sap(real_trajectory_encode_h_list[:, v, :])
                label = input_y_train_[:, v]
                label = label.reshape((-1, 1)).astype('float32')
                clf_loss += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predicted_output))
                neg_likelihood_loss += partial_log_likelihood(predicted_output, input_t_train_[:, v], label)
            survival_prediction_loss = tf.add(clf_loss, neg_likelihood_loss)

            if SHUFFLE_RATE==0:
                shuffled_input_x_train=input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]

                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1

            # shuffled_input_x_train = input_x_train_[:, shuffle_index, :]

            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder((shuffled_input_x_train),
                                                                                    batch=batch_size)

            # contrast_loss_matrix = tf.matmul(mlp(s_h), tf.transpose(mlp(r_h)))
            # contrast_loss_matrix = -tf.keras.losses.cosine_similarity(tf.expand_dims(mlp(s_h), 1),
            #                                                           tf.expand_dims(mlp(r_h), 0))
            # 对比学习
            contrast_loss_matrix = tf.matmul(mlp(shuffled_generated_decode_h), tf.transpose(mlp(real_decode_h)))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1) - contrast_loss_numerator

            contrast_loss_matrix2 = tf.matmul(mlp(real_decode_h), tf.transpose(mlp(shuffled_generated_decode_h)))
            contrast_loss_numerator2 = tf.linalg.diag_part(contrast_loss_matrix2)
            contrast_loss_denominator2 = tf.reduce_sum(tf.math.exp(contrast_loss_matrix2),
                                                      axis=1) - contrast_loss_numerator2

            contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))-tf.reduce_mean(contrast_loss_numerator2 - tf.math.log(contrast_loss_denominator2))
            whole_loss = clf_loss + gen_mse_loss * 0.3 + neg_likelihood_loss * 1 + contrast_loss * 0.01
            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            sap_variables = [var for var in sap.trainable_variables]
            for weight in sap.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            mlp_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            variables = sap_variables + encoder_variables + decoder_variables + mlp_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            # if train_set.epoch_completed == 1:
            #     s2s.load_weights('S2S_weight_{}_v1.h5'.format(target_label))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                input_day_test = test_set.day
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder((input_x_test), batch=batch_test)
                predicted_output_test_list = np.zeros((batch_test, 0))
                for v in range(visit_len):
                    predicted_output_test = sap(real_trajectory_encode_h_list_test[:, v, :]).numpy()
                    predicted_output_test_list = np.concatenate([predicted_output_test_list, predicted_output_test],
                                                                axis=1)

                label_test = input_y_test.reshape((-1, 1)).astype('float32')
                predicted_output_test_list = predicted_output_test_list.reshape((-1, 1))
                try:
                    y_pred_label, auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(
                        label_test, predicted_output_test_list)
                    c_index = concordance_index(input_t_test.reshape((-1, 1)), -predicted_output_test_list,
                                                label_test)
                    result.append([c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test])

                except:
                    break

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print(
                        '----epoch:{}, whole_loss:{}, contrast_loss:{},survaval_loss:{},gen_loss:{}, c_index:{},auc:{},precision_test:{},recall_test:{},f_score_test:{},accuracy_test:{}, '
                        '-count:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss, clf_loss+neg_likelihood_loss,
                                           gen_mse_loss, c_index,
                                           auc_test, precision_test, recall_test, f_score_test, accuracy_test, count))

        # if train_set.epoch_completed == epochs - 1:
        #     encoder.save_weights('save/train_model_test_encoder_weight_{}_mimic.h5'.format(target_label))
        #     decoder.save_weights('save/train_model_test_decoder_weight_{}_mimic.h5'.format(target_label))
        #     sap.save_weights('save/train_model_test_sap_weight_{}_mimic.h5'.format(target_label))

        tf.compat.v1.reset_default_graph()
    result = np.array(result)
    max_i = np.where(result[:, 0] == result[:, 0].max())
    print(np.squeeze(result[max_i, :]))
    return np.squeeze(result[max_i, :])
def train_model_mimic_timelstm3(hidden_size, learning_rate, l2_regularization, time_windows):
    """
    timelstm3

    :param hidden_size:hidden_size
    :param learning_rate:learning_rate
    :param l2_regularization:l2_regularization
    :param time_windows:prediction survival time windows
    :return:[c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test]
    """

    target_label = time_windows
    train_set, test_set = read_mimic_data_with_time_interval()
    feature = train_set.x
    feature_dims = feature.shape[2]
    visit_len = feature.shape[1]

    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    # shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)

    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))
    # discriminator = DISCRIMINATOR(hidden_size=hidden_size, feature_dims=28)
    encoder = Encoder(hidden_size=32, model_type='LSTM')
    decoder = Decoder(hidden_size=32, feature_dims=feature_dims, model_type='TimeLSTM3')
    sap = SAP(hidden_size=256)
    mlp = MLP2(hidden_size=64)
    logged = set()
    max_loss = 0.01
    max_pace = 0.0001
    result = []
    shuffle_index = [2,1,0]
    count = 0
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_, input_t_train_, input_y_train_, input_day_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 做mask
            # mask_index = int(np.random.random() * visit_len)
            mask_index = int(np.random.random() * (visit_len-1))

            if MASK_RATE==0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:

                # mask_index = 2
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select = np.random.random()
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            # 生成预测的序列
            trajectory_encode_last_h, trajectory_encode_h_list = encoder((mask_input_x_train), batch=batch_size)
            # context_state = mlp(trajectory_encode_last_h)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder((trajectory_encode_last_h,input_day_train_),
                                                                                  predicted_visit=visit_len,
                                                                                  batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))

            real_decode_h, real_trajectory_encode_h_list = encoder((input_x_train_), batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            for v in range(visit_len):
                predicted_output = sap(real_trajectory_encode_h_list[:, v, :])
                label = input_y_train_[:, v]
                label = label.reshape((-1, 1)).astype('float32')
                clf_loss += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predicted_output))
                neg_likelihood_loss += partial_log_likelihood(predicted_output, input_t_train_[:, v], label)
            survival_prediction_loss = tf.add(clf_loss, neg_likelihood_loss)

            if SHUFFLE_RATE==0:
                shuffled_input_x_train=input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]

                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1

            # shuffled_input_x_train = input_x_train_[:, shuffle_index, :]

            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder((shuffled_input_x_train),
                                                                                    batch=batch_size)

            # contrast_loss_matrix = tf.matmul(mlp(s_h), tf.transpose(mlp(r_h)))
            # contrast_loss_matrix = -tf.keras.losses.cosine_similarity(tf.expand_dims(mlp(s_h), 1),
            #                                                           tf.expand_dims(mlp(r_h), 0))
            # 对比学习
            contrast_loss_matrix = tf.matmul(mlp(shuffled_generated_decode_h), tf.transpose(mlp(real_decode_h)))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1) - contrast_loss_numerator

            contrast_loss_matrix2 = tf.matmul(mlp(real_decode_h), tf.transpose(mlp(shuffled_generated_decode_h)))
            contrast_loss_numerator2 = tf.linalg.diag_part(contrast_loss_matrix2)
            contrast_loss_denominator2 = tf.reduce_sum(tf.math.exp(contrast_loss_matrix2),
                                                      axis=1) - contrast_loss_numerator2

            contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))-tf.reduce_mean(contrast_loss_numerator2 - tf.math.log(contrast_loss_denominator2))
            whole_loss = clf_loss + gen_mse_loss * 0.3 + neg_likelihood_loss * 1 + contrast_loss * 0.01
            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            sap_variables = [var for var in sap.trainable_variables]
            for weight in sap.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            mlp_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            variables = sap_variables + encoder_variables + decoder_variables + mlp_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            # if train_set.epoch_completed == 1:
            #     s2s.load_weights('S2S_weight_{}_v1.h5'.format(target_label))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                input_day_test = test_set.day
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder((input_x_test), batch=batch_test)
                predicted_output_test_list = np.zeros((batch_test, 0))
                for v in range(visit_len):
                    predicted_output_test = sap(real_trajectory_encode_h_list_test[:, v, :]).numpy()
                    predicted_output_test_list = np.concatenate([predicted_output_test_list, predicted_output_test],
                                                                axis=1)

                label_test = input_y_test.reshape((-1, 1)).astype('float32')
                predicted_output_test_list = predicted_output_test_list.reshape((-1, 1))
                try:
                    y_pred_label, auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(
                        label_test, predicted_output_test_list)
                    c_index = concordance_index(input_t_test.reshape((-1, 1)), -predicted_output_test_list,
                                                label_test)
                    result.append([c_index, auc_test, precision_test, recall_test, f_score_test, accuracy_test])

                except:
                    break

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print(
                        '----epoch:{}, whole_loss:{}, contrast_loss:{},survaval_loss:{},gen_loss:{}, c_index:{},auc:{},precision_test:{},recall_test:{},f_score_test:{},accuracy_test:{}, '
                        '-count:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss, clf_loss+neg_likelihood_loss,
                                           gen_mse_loss, c_index,
                                           auc_test, precision_test, recall_test, f_score_test, accuracy_test, count))

        # if train_set.epoch_completed == epochs - 1:
        #     encoder.save_weights('save/train_model_test_encoder_weight_{}_mimic.h5'.format(target_label))
        #     decoder.save_weights('save/train_model_test_decoder_weight_{}_mimic.h5'.format(target_label))
        #     sap.save_weights('save/train_model_test_sap_weight_{}_mimic.h5'.format(target_label))

        tf.compat.v1.reset_default_graph()
    result = np.array(result)
    max_i = np.where(result[:, 0] == result[:, 0].max())
    print(np.squeeze(result[max_i, :]))
    return np.squeeze(result[max_i, :])


def train_and_save():
    auc_all = []
    c_index_all = []
    result_all = []
    mse_all = []
    mae_all = []
    r_value_all = []
    p_value_all = []

    hidden_size = 64
    learning_rate = 0.001
    l2_regularization = 0.0001
    reapet = 50
    model_type = 'train_model_mimic_all'
    # time_windows = 'half_year'
    # time_windows = 'one_year'
    time_windows = 'three_days'
    # time_windows = 'six_month'
    print('model type:{}'.format(model_type))
    for i in range(reapet):

        if model_type == 'train_model_mimic_all':
            # RNN
            c_index, auc, precision, recall, f_score, accuracy = train_model_mimic_all(
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                time_windows=time_windows)

        elif model_type == 'train_model_mimic_timelstm1':
            c_index, auc, precision, recall, f_score, accuracy = train_model_mimic_timelstm1(
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                time_windows=time_windows)
        elif model_type == 'train_model_mimic_timelstm2':
            c_index, auc, precision, recall, f_score, accuracy = train_model_mimic_timelstm1(
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                time_windows=time_windows)
        elif model_type == 'train_model_mimic_timelstm3':
            c_index, auc, precision, recall, f_score, accuracy = train_model_mimic_timelstm1(
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                time_windows=time_windows)
        else:
            c_index, auc, precision, recall, f_score, accuracy = train_model_mimic_timelstm1(
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization)

        auc_all.append(auc)
        c_index_all.append(c_index)
        result_all.append([c_index, auc, precision, recall, f_score, accuracy])

        print('epoch  {}-----auc-all_ave  {}-----c_index_all-all_ave  {}----mae_all_ave-----{}---r_value_ave  {}--'
              '---p_value_ave  {}--  mse_vale_std{}------mae_vale_std{}---r_value_std{}  p_value_std-'.
              format(i, np.mean(auc_all), np.mean(c_index_all), np.mean(mae_all),
                     np.mean(r_value_all), np.mean(p_value_all),
                     np.std(mse_all), np.std(mae_all),
                     np.std(r_value_all), np.std(p_value_all)))
        # 对比学习的目的在于更好的半监督学习
        # fro
        if i % 5 == 0 and i > 0:
            re_save = pd.DataFrame(result_all, columns=['c_index', 'auc', 'precision', 'recall', 'f_score', 'accuracy'])
            re_save.to_excel(
                '{}/newsh_rate={}_mask_RATE={}_{}_{}_hidden_size={}_learning_rate={}_l2_regularization={}_reapet{}times.xlsx'.format(
                    RESULT_SAVE_DIR,SHUFFLE_RATE,MASK_RATE, model_type, time_windows, hidden_size, learning_rate, l2_regularization, reapet))
    result_all = pd.DataFrame(result_all, columns=['c_index', 'auc', 'precision', 'recall', 'f_score', 'accuracy'])
    result_all.to_excel(
        '{}/newsh_rate={}_mask_RATE={}_{}_{}_hidden_size={}_learning_rate={}_l2_regularization={}_reapet{}times5_v3.xlsx'.format(RESULT_SAVE_DIR,SHUFFLE_RATE, MASK_RATE,
                                                                                                   model_type,
                                                                                                   time_windows,
                                                                                                   hidden_size,
                                                                                                   learning_rate,
                                                                                                   l2_regularization,
                                                                                                   reapet,
                                                                                                   ))
if __name__ == '__main__':




    # 预训练
    # test_test('输出记录.txt')

    # Encode_Decode_Time_BO = BayesianOptimization(
    #     search_model_mimic, {
    #         'hidden_size1': (4, 8),
    #         'hidden_size2': (4, 8),
    #         'hidden_size3': (4, 8),
    #         'hidden_size4': (4, 8),
    #         'a1': (0.01, 10),
    #         'a2': (0.01, 10),
    #         'a3': (0.01, 10),
    #         'learning_rate': (-4, -2),
    #         'l2_regularization': (-5, -1),
    #
    #     }
    # )
    # Encode_Decode_Time_BO.maximize(n_iter=50)
    # # print('plagh half year')
    # print('mimic')
    # # print(MAX_CINDEX,hidden_size1_final,hidden_size2_final,hidden_size3_final,learning_rate_final, l2_regularization_final)
    # print(MAX_CINDEX, hidden_size1_final, hidden_size2_final, hidden_size3_final, hidden_size4_final, a1_final,a2_final,a3_final, learning_rate_final,
    #       l2_regularization_final)
    # print(Encode_Decode_Time_BO.max)
    train_and_save()






