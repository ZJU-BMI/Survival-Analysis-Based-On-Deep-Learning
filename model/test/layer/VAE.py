from bayes_opt import BayesianOptimization
import scipy.stats as stats
import os
import sys
import numpy as np
# from utils import split_dataset
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import *
from tensorflow import keras
from test.RNN import RNN
from test.output_layer import *
from utils import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve

class VAE(tf.keras.Model):

    def __init__(self, hidden_size, feature_dims, z_dims, previous_visit, predicted_visit):
        super(VAE, self).__init__(name='VAE')

        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit

        # 首先定义各种layer层的各部分
        # ENCODER
        self.hidden_size = hidden_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size, implementation=2)

        # DECODER
        self.feature_dims = feature_dims
        self.LSTM_Cell_decoder = tf.keras.layers.LSTMCell(hidden_size, implementation=2)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

        # Prior
        self.z_dims = z_dims
        self.prior_dense1 = tf.keras.layers.Dense(z_dims)
        self.prior_dense2 = tf.keras.layers.Dense(z_dims)
        self.prior_dense3 = tf.keras.layers.Dense(z_dims)  # obtain hidden z
        self.prior_dense4 = tf.keras.layers.Dense(z_dims)
        self.prior_dense5 = tf.keras.layers.Dense(z_dims)

        # Post
        self.post_dense1 = tf.keras.layers.Dense(z_dims)
        self.post_dense2 = tf.keras.layers.Dense(z_dims)
        self.post_dense3 = tf.keras.layers.Dense(z_dims)
        # obtain z_mean, z_log_var
        self.post_dense4 = tf.keras.layers.Dense(z_dims)
        self.post_dense5 = tf.keras.layers.Dense(z_dims)

    # ENCODER
    def encode(self, input_x):
        sequence_time, h, c = input_x
        output, state = self.LSTM_Cell_encode(sequence_time, [h, c])

        return state[0], state[1]

    # DECODER
    def decode(self, input_x):
        z_i_1, h_i, x_i, decode_c, decode_h = input_x
        input_decode = tf.concat((z_i_1, h_i, x_i), axis=1)
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decoder(input_decode, state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]

    def prior(self, prior_input):
        hidden_1 = self.prior_dense1(prior_input)
        hidden_2 = self.prior_dense2(hidden_1)
        hidden_3 = self.prior_dense3(hidden_2)
        z_mean = self.prior_dense4(hidden_3)
        z_log_var = self.prior_dense5(hidden_3)
        z = self.reparameterize(z_mean, z_log_var, self.z_dims)
        return z, z_mean, z_log_var

    def post(self, post_input):
        h_i, h_i_1 = post_input
        hidden = tf.concat((h_i, h_i_1), axis=1)
        hidden_1 = self.post_dense1(hidden)
        hidden_2 = self.post_dense2(hidden_1)
        hidden_3 = self.post_dense3(hidden_2)
        z_mean = self.post_dense4(hidden_3)
        z_log_var = self.post_dense5(hidden_3)
        z = self.reparameterize(z_mean, z_log_var, self.z_dims)
        return z, z_mean, z_log_var

    @staticmethod
    def reparameterize(mu, log_var, z_dims):
        batch = mu.shape[0]
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        return z

    # 最后在call中整合并输出
    def call(self, input_x_train, batch=0, train_flag=True, training=None, mask=None):
        # 输入： 真实序列
        # 输出： 生成序列和重建序列，几个损失函数等

        input_x_train_ = input_x_train[:, 0:self.previous_visit, :]
        generated_trajectory = np.zeros(shape=[batch, 0, self.feature_dims])
        construct_trajectory = np.zeros(shape=[batch, 0, self.feature_dims])
        z_log_var_post_all = np.zeros(shape=[batch, 0, self.z_dims])
        z_mean_post_all = np.zeros(shape=[batch, 0, self.z_dims])
        z_log_var_prior_all = np.zeros(shape=[batch, 0, self.z_dims])
        z_mean_prior_all = np.zeros(shape=[batch, 0, self.z_dims])
        decode_c_generate = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h_generate = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_c_reconstruct = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h_reconstruct = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        for predicted_visit_ in range(self.predicted_visit):
            if train_flag == False and predicted_visit_>0:
                if predicted_visit_==1:
                    sequence_last_time = input_x_train[:, self.previous_visit, :]

                else:
                    sequence_last_time = generated_trajectory[:, predicted_visit_-1, :]

            else:
                sequence_last_time = input_x_train[:, predicted_visit_ + self.previous_visit - 1, :]
            sequence_time_current_time = input_x_train[:, predicted_visit_ + self.previous_visit, :]
            encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            for previous_visit_ in range(self.previous_visit + predicted_visit_):
                if train_flag:
                    sequence_time = input_x_train[:, previous_visit_, :]
                else:
                    sequence_time = input_x_train_[:, previous_visit_, :]
                encode_c, encode_h = self.encode([sequence_time, encode_c, encode_h])

            context_state = encode_h
            z_prior, z_mean_prior, z_log_var_prior = self.prior(context_state)  # h_i--> z_(i+1)
            encode_c, encode_h = self.encode([sequence_time_current_time, encode_c, encode_h])  # h_(i+1)
            z_post, z_mean_post, z_log_var_post = self.post([context_state, encode_h])  # h_i, h_(i+1) --> z_(i+1)
            construct_next_visit, decode_c_reconstruct, decode_h_reconstruct = self.decode(
                [z_post, context_state, sequence_last_time, decode_c_reconstruct, decode_h_reconstruct])
            construct_next_visit = tf.reshape(construct_next_visit, [batch, -1, self.feature_dims])
            # 重建的序列
            construct_trajectory = tf.concat((construct_trajectory, construct_next_visit), axis=1)
            generated_next_visit, decode_c_generate, decode_h_generate = self.decode(
                [z_prior, context_state, sequence_last_time, decode_c_generate, decode_h_generate])
            generated_next_visit = tf.reshape(generated_next_visit, (batch, -1, self.feature_dims))
            # 生成的序列
            generated_trajectory = tf.concat((generated_trajectory, generated_next_visit), axis=1)
            input_x_train_ = tf.concat((input_x_train_, generated_next_visit), axis=1)
            z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, self.z_dims])), axis=1)
            z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, self.z_dims])), axis=1)
            z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, self.z_dims])),
                                            axis=1)
            z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, self.z_dims])),
                                           axis=1)

        return construct_trajectory, generated_trajectory, z_mean_prior_all, z_mean_post_all, z_log_var_prior_all, z_log_var_post_all


def train_vae(hidden_size, z_dims, l2_regularization, learning_rate, kl_imbalance, reconstruction_imbalance, generated_mse_imbalance, survival_prediction_imbalance):
    """
    LSTM + ALL LOSS

    :param hidden_size:
    :param learning_rate:
    :param l2_regularization:
    :return:
    """

    target_label = 'two_year'

    with open('data/test_set_{}.pkl'.format(target_label), 'rb') as f:
        test_set = pickle.load(f)
    with open('data/train_set_{}.pkl'.format(target_label), 'rb') as f:
        train_set = pickle.load(f)
    previous_visit = 2
    predicted_visit = 3
    feature_dims = 28
    epochs = 300
    batch_size = 128
    #
    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)

    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # z_dims = 2 ** (int(z_dims))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # kl_imbalance = 10 ** kl_imbalance
    # reconstruction_imbalance = 10 ** reconstruction_imbalance
    # generated_mse_imbalance = 10 ** generated_mse_imbalance
    # survival_prediction_imbalance = 10 ** survival_prediction_imbalance
    print('hidden_size{}----z_dims{}------learning_rate{}----l2_regularization{}---'
          'kl_imbalance{}----reconstruction_imbalance '
          ' {}----generated_mse_imbalance{}----'.format(hidden_size, z_dims,
                                                        learning_rate,
                                                        l2_regularization,
                                                        kl_imbalance,
                                                        reconstruction_imbalance,
                                                        generated_mse_imbalance))
    discriminator = DISCRIMINATOR(hidden_size=hidden_size, feature_dims=feature_dims)
    vae = VAE(hidden_size, feature_dims, z_dims, previous_visit, predicted_visit)
    sap = SAP(hidden_size=hidden_size, feature_dims=feature_dims)
    encoder = ENCODER(hidden_size=hidden_size)
    logged = set()
    max_loss = 0.01
    max_pace = 0.0001

    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_,input_t_train_, input_y_train_ = train_set.next_batch(batch_size)

        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 生成预测的序列
            construct_trajectory, generated_trajectory, z_mean_prior_all, z_mean_post_all, z_log_var_prior_all, z_log_var_post_all = vae(input_x_train_, batch=batch_size, train_flag=True)
            gen_mse_loss = tf.reduce_mean(tf.keras.losses.mse(
                input_x_train_[:, previous_visit: previous_visit + predicted_visit, :], generated_trajectory))
            # 合并
            input_x_train = np.concatenate((input_x_train_[:, 0:previous_visit, :], generated_trajectory), axis=1)
            # 输出表征
            generated_decode_h = encoder(input_x_train, batch=batch_size)

            real_decode_h = encoder(input_x_train_, batch=batch_size)

            # 对比损失
            # contrast_loss_matrix = tf.matmul(generated_decode_h, tf.transpose(real_decode_h))
            # contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix), axis=1)
            # contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            # contrast_loss = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))

            # VAE损失
            reconstruction_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    construct_trajectory))


            std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
            std_prior = tf.math.sqrt(tf.exp(z_mean_prior_all))
            kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(tf.maximum(std_post,
                                                                                                               1e-9)) +
                                     (tf.math.pow(std_post, 2) + tf.math.pow((z_mean_post_all - z_mean_prior_all), 2)) /
                                     tf.maximum(tf.math.pow(std_prior, 2), 1e-9) - 1)
            kl_loss_all = tf.reduce_mean(kl_loss_element)
            # 生存分析损失
            clf_loss = 0
            neg_likelihood_loss = 0


            for v in range(predicted_visit):
                input_trajectory = tf.concat(
                    (input_x_train_[:, 0:previous_visit + v, :], generated_trajectory[:, v:v + 1, :]), axis=1)
                predicted_output = sap(encoder(input_trajectory, batch=batch_size))
                label = input_y_train_[:, previous_visit+v, :]
                label = label.reshape((-1, 1)).astype('float32')
                clf_loss = clf_loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predicted_output))
                neg_likelihood_loss = neg_likelihood_loss + partial_log_likelihood(predicted_output,
                                                                                   input_t_train_[:, previous_visit + v,
                                                                                   :], label)
            survival_prediction_loss = tf.add(clf_loss, neg_likelihood_loss)
            whole_loss = reconstruction_mse_loss * reconstruction_imbalance + kl_loss_all * kl_imbalance + gen_mse_loss * generated_mse_imbalance + survival_prediction_loss * survival_prediction_imbalance
            # whole_loss = survival_prediction_loss + contrast_loss + discriminator_loss + gen_mse_loss + rec_mse_loss*0
            # whole_loss = survival_prediction_loss + 200 * gen_mse_loss
            vae_variables = [var for var in vae.trainable_variables]
            for weight in vae.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            #
            sap_variables = [var for var in sap.trainable_variables]
            for weight in sap.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            discriminator_variables = [var for var in discriminator.trainable_variables]
            for weight in discriminator.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            # if train_set.epoch_completed == 0:
            #     # encoder_decoder.load_weights('RNN_weight.h5')
            #     s2s.load_weights('S2S_weight_{}.h5'.format(target_label))
            variables = vae_variables + sap_variables + encoder_variables + discriminator_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))
            # if train_set.epoch_completed == 1:
            #     s2s.load_weights('S2S_weight_{}_v1.h5'.format(target_label))
            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                batch_test = input_x_test.shape[0]
                construct_trajectory_test, generated_trajectory_test, z_mean_prior_all_test, z_mean_post_all_test, z_log_var_prior_all_test, z_log_var_post_all_test = vae(input_x_test, batch=batch_test,
                                                                           train_flag=False)

                mse_loss_predicted_test = tf.reduce_mean(
                    tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit + predicted_visit, :],
                                        generated_trajectory_test)).numpy()
                mae_predicted_test = tf.reduce_mean(
                    tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :],
                                        generated_trajectory_test)).numpy()
                input_trajectory_test = tf.concat(
                    (input_x_test[:, 0:previous_visit, :], generated_trajectory_test), axis=1)

                for v in range(predicted_visit):
                    predicted_output_test = sap(
                        encoder(input_trajectory_test[:, 0:previous_visit + v + 1, :], batch=batch_test))
                    if v == 0:
                        predicted_output_list_test = predicted_output_test
                    else:
                        predicted_output_list_test = np.concatenate((predicted_output_list_test, predicted_output_test), axis=1)
                predicted_output_list_test = np.reshape(predicted_output_list_test, (-1, 1))
                y_label_test = np.reshape(input_y_test[:, previous_visit:, :], (-1, 1))

                auc_test, precision_test, recall_test, f_score_test, accuracy_test = calculate_score(y_label_test, predicted_output_list_test)


                r_value_all = []
                p_value_all = []
                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print('----epoch:{}, whole_loss:{}, reconstruction_mse_loss:{},kl_loss_all:{},neg_likelihood_loss:{},predicted_mse:{}, mae_predicted:{}, auc:{}, '
                          'predicted_r_value:{}--count:{}'.format(train_set.epoch_completed,
                                                                          whole_loss,reconstruction_mse_loss,kl_loss_all, neg_likelihood_loss, mse_loss_predicted_test,
                                                                          mae_predicted_test,auc_test,
                                                                          np.mean(r_value_all), count))

                # if (np.mean(r_value_all) > 0.87) and (np.mean(r_value_all) < 0.88) and (
                #         train_set.epoch_completed == 49):
                #     np.savetxt('AED_generated_trajectory.csv',
                #                predicted_trajectory_test.numpy().reshape(-1, feature_dims), delimiter=',')
        # if train_set.epoch_completed == epochs-1:
        #     s2s.save_weights('S2S_weight_{}.h5'.format(target_label))

        tf.compat.v1.reset_default_graph()

    return auc_test, precision_test, recall_test, f_score_test, accuracy_test, mse_loss_predicted_test, mae_predicted_test, np.mean(r_value_all), np.mean(p_value_all)
    # return auc_test - mse_loss_predicted_test*100


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


if __name__ == '__main__':

    target_label = 'two_year'

    # 预训练
    # test_test('AED_—HF_test__3_3_5_重新训练.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train_vae, {
    #         'hidden_size': (5, 8),
    #         'z_dims': (5, 8),
    #         'learning_rate': (-5, -3),
    #         'l2_regularization': (-5, -1),
    #         'kl_imbalance': (-2, 2),
    #         'reconstruction_imbalance': (-2, 2),
    #         'generated_mse_imbalance': (-2, 2),
    #         'survival_prediction_imbalance': (-2, 2),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize(n_iter=50)
    # print(Encode_Decode_Time_BO.max)
    auc_all =[]
    result_all = []
    mse_all = []
    mae_all = []
    r_value_all = []
    p_value_all = []

    hidden_size = 64
    learning_rate = 0.001
    l2_regularization = 2.92e-05
    reapet = 50
    model_type = 'VAE'
    print('model type:{}'.format(model_type))
    for i in range(reapet):

        if model_type == 'VAE':
            # RNN
            auc, precision, recall, f_score, accuracy, mse, mae, r_value, p_value = train_vae(hidden_size=64,
                                  learning_rate=0.001,
                                  l2_regularization=1.e-05,
                                  z_dims=64,
                                  kl_imbalance=0.01,
                                  generated_mse_imbalance=100,
                                  reconstruction_imbalance=0.01,
                                  survival_prediction_imbalance=1)

        else:
            auc, precision, recall, f_score, accuracy, mse, mae, r_value, p_value = train_vae(hidden_size=32,
                                  learning_rate=0.005686519630243845,
                                  l2_regularization=1.0101545386867363e-05,
                                  z_dims=32,
                                  kl_imbalance=4.042659336415265,
                                  generated_mse_imbalance=0.04670594856700065,
                                  reconstruction_imbalance=0.00019355771673396988)

        mse_all.append(mse)
        r_value_all.append(r_value)
        p_value_all.append(p_value)
        mae_all.append(mae)
        auc_all.append(auc)
        result_all.append([auc, precision, recall, f_score, accuracy, mse, mae, r_value, p_value])

        print('epoch  {}-----auc-all_ave  {}-----mse-all_ave  {}----mae_all_ave-----{}---r_value_ave  {}--'
              '---p_value_ave  {}--  mse_vale_std{}------mae_vale_std{}---r_value_std{}  p_value_std-'.
              format(i, np.mean(auc_all), np.mean(mse_all), np.mean(mae_all),
                     np.mean(r_value_all), np.mean(p_value_all),
                     np.std(mse_all), np.std(mae_all),
                     np.std(r_value_all), np.std(p_value_all)))
        if i % 5 == 0 and i > 0:
            re_save = pd.DataFrame(result_all, columns=['auc', 'precision', 'recall', 'f_score', 'accuracy', 'mse', 'mae', 'r_value', 'p_value'])
            re_save.to_excel('{}/newencoder1125_{}_{}_hidden_size={}_learning_rate={}_l2_regularization={}_reapet{}times.xlsx'.format(RESULT_SAVE_DIR, model_type, target_label, hidden_size, learning_rate, l2_regularization,reapet))
    result_all = pd.DataFrame(result_all, columns=['auc', 'precision', 'recall', 'f_score', 'accuracy', 'mse', 'mae', 'r_value', 'p_value'])
    result_all.to_excel('{}/newencoder1125_{}_{}_hidden_size={}_learning_rate={}_l2_regularization={}_reapet{}times.xlsx'.format(RESULT_SAVE_DIR, model_type, target_label, hidden_size, learning_rate, l2_regularization,reapet))