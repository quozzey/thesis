import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Flatten, Dense, Input, Lambda, Dropout, Softmax
from keras import Model
import keras.backend as K
from keras.optimizers import Adam

np.random.seed(123)
tf.random.set_seed(123)


def softmax_mod(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1) / tf.expand_dims(
        tf.reduce_sum(tf.exp(tf.abs(x)) - 1, axis=1), axis=-1)


class Trade_Model:
    def __init__(self, n_assets, batch_size, t_cost=0.0005):
        self.data = None
        self.model = None
        self.price_data_train = None
        self.price_data_val = None
        self.n_assets = n_assets
        self.batch_size = batch_size
        self.t_cost = t_cost

    def __build_model(self, input_shape, outputs):

        inp = Input(shape=input_shape)
        l1 = LSTM(64, input_shape=input_shape, return_sequences=True)(inp)
        l1 = LSTM(64, input_shape=input_shape, return_sequences=False)(l1)
        l1 = Dropout(0.2)(l1)
        #l1 = Dropout(0.05)(l1)
        l1 = Flatten()(l1)
        l1 = Dense(outputs)(l1)
        #outp = Softmax()(l1)
        outp = Lambda(softmax_mod(l1))
        model = Model(inputs=inp, outputs=outp)

        def sharpe_loss(returns, y_pred):
            portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
            costs = tf.concat([tf.zeros((1, )), self.t_cost * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
            portfolio_returns = portfolio_returns - costs

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns) * np.sqrt(252)
            return -sharpe

        def soft_target_simple(returns, y_pred):
            portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
            costs = tf.concat([tf.zeros((1, )), self.t_cost * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
            portfolio_returns = portfolio_returns - costs
            volat = K.std(portfolio_returns) * np.sqrt(252)
            ret = (K.exp(tf.reduce_mean(K.log(1+portfolio_returns), axis=0)) ** 252 - 1)
            return volat - K.min([ret, 0])

        model.compile(loss=soft_target_simple, optimizer=Adam(learning_rate=1e-5), run_eagerly=False)
        return model

    def get_allocations(self, data, price_data):
        data_train = data

        price_data_train = price_data

        self.price_data_train = tf.cast(tf.constant(price_data_train), float)

        if self.model is None:
            self.model = self.__build_model((data.shape[1], data.shape[2]), self.n_assets)

        if self.batch_size is None:
            self.model.fit(data_train[:-1, :, :], self.price_data_train[1:] / self.price_data_train[:-1] - 1,
                           epochs=100,
                           shuffle=False, batch_size=data_train.shape[0] - 1, validation_split=0.1)

        else:
            if 0.9 * (data_train.shape[0] - 1) % self.batch_size < 3:
                self.batch_size += 1

            self.model.fit(data_train[:-1, :, :], self.price_data_train[1:] / self.price_data_train[:-1] - 1,
                           epochs=100,
                           shuffle=False, batch_size=self.batch_size, validation_split=0.1)
