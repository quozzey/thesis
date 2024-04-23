import numpy as np

# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from keras.layers import LSTM, Flatten, Dense, Input, Lambda, Dropout, Multiply
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.activations import relu
from keras.regularizers import L1, L2


def scheduler(epoch, lr):
    if epoch < 10:
        return 0.001
    else:
        return 0.00003


def softmax_mod(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1) / tf.expand_dims(
        tf.reduce_sum(tf.exp(tf.abs(x)) - 1, axis=1), axis=-1)


def softmax_mod_naive(x):
    return tf.sign(x) * tf.exp(tf.abs(x)) / tf.expand_dims(tf.reduce_sum(tf.exp(tf.abs(x)), axis=1), axis=-1)


def softmax_mod2(x):
    return tf.sign(x) * (tf.exp(x ** 2) - 1) / tf.expand_dims(tf.reduce_sum(tf.exp(x ** 2) - 1, axis=1), axis=-1)


def tanh_mod(x):
    return tf.tanh(x) / tf.expand_dims(tf.reduce_sum(tf.abs(tf.tanh(x)), axis=1), axis=-1)


def double_relu(x):
    return relu(x, threshold=0.5) - relu(-x, threshold=0.5) + 1e-4


def double_relu_norm(x):
    return double_relu(x) / tf.expand_dims(tf.reduce_sum(tf.abs(double_relu(x)), axis=1), axis=-1)


class Trade_Model:
    def __init__(self, n_assets, loss, batch_size=64):
        self.data = None
        self.model = None
        self.price_data_train = None
        self.price_data_val = None
        self.n_assets = n_assets
        self.batch_size = batch_size
        self.loss = loss

    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''

        # Model A
        # inp = Input(shape=input_shape)
        # l1 = LSTM(outputs, input_shape=input_shape, return_sequences=False)(inp)
        # l1 = Flatten()(l1)
        # l1 = Dense(outputs)(l1)

        # s1 = LSTM(outputs, input_shape=input_shape, return_sequences=False)(inp)
        # s1 = Flatten()(s1)
        # s1 = Dense(outputs)(s1)

        # l2 = Softmax()(l1 - s1)
        # s2 = Softmax()(s1 - l1)

        # model = Model(inputs=inp, outputs=l2 - s2)

        # Model B

        # model = Sequential([
        #    #LSTM(256, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.1),
        #    LSTM(64, input_shape=input_shape, return_sequences=False),
        #    #LSTM(32, input_shape=input_shape, return_sequences=False),
        #    Flatten(),
        #    Dense(outputs, activation=softmax_mod),
        # ])

        # MODEL C
        inp = Input(shape=input_shape)
        l1 = LSTM(128, input_shape=input_shape, return_sequences=True)(inp)
        l1 = Dropout(0.05)(l1)
        l1 = LSTM(64, input_shape=input_shape, return_sequences=False)(l1)
        l1 = Flatten()(l1)
        l1 = Dropout(0.05)(l1)
        #l1 = Dropout(0.05)(l1)

        outp = Dense(outputs, activation=softmax_mod)(l1)
        # l1 = Lambda(lambda x: tf.abs(x))(s1)
        # l1 = Softmax()(l1)
        # s1 = Lambda(lambda x: tf.sign(x))(s1)
        model = Model(inputs=inp, outputs=outp)

        def sharpe_loss(returns, y_pred):
            # make all time-series start at 1
            # if y_pred.shape[0] == self.price_data_train.shape[0] - 1:
            # sums = tf.reduce_sum(tf.abs(y_pred), axis=1)
            # y_pred = tf.divide(y_pred, K.maximum(tf.expand_dims(sums, axis=-1), 1))
            # data = tf.divide(self.price_data_train, self.price_data_train[day[0, 0]])
            # elif y_pred.shape[0] == self.price_data_val.shape[0] - 1:
            #    data = tf.divide(self.price_data_val, self.price_data_val[0])
            # else:
            #    print('Incorrect shape in loss!')
            #    print(y_pred.shape)
            #    print(self.price_data_train.shape)
            #    print(self.price_data_val.shape)

            # value of the portfolio after allocations applied

            portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
            costs = tf.concat([tf.zeros((1, )), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
            portfolio_returns = portfolio_returns - costs
            ret = (K.exp(tf.reduce_mean(K.log(1 + portfolio_returns), axis=0)) ** 252 - 1)

            sharpe = ret / K.std(portfolio_returns) * np.sqrt(252)

            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe


        def soft_target_simple(returns, y_pred):

            portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
            costs = tf.concat([tf.zeros((1, )), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
            portfolio_returns = portfolio_returns - costs
            volat = K.std(portfolio_returns) * np.sqrt(252)
            ret = (K.exp(tf.reduce_mean(K.log(1+portfolio_returns), axis=0)) ** 252 - 1)



            # since we want to maximize Sharpe, while gradient descent minimizes the loss,
            #   we can negate Sharpe (the min of a negated function is its max)
            return volat - K.min([ret, 0])


        model.compile(loss=self.loss, optimizer=Adam(learning_rate=5e-6), run_eagerly=False)
        return model


    def get_allocations(self, data, price_data):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''

        data_train = data

        price_data_train = price_data

        self.price_data_train = tf.cast(tf.constant(price_data_train), float)

        if self.model is None:
            self.model = self.__build_model((data.shape[1], data.shape[2]), self.n_assets)

        stopping = EarlyStopping('val_loss', patience=5, restore_best_weights=True, start_from_epoch=50)
        schedule = LearningRateScheduler(scheduler)

        self.model.fit(data_train[:-1, :, :], self.price_data_train[1:] / self.price_data_train[:-1] - 1, epochs=100,
                       shuffle=False, batch_size=self.batch_size, validation_split=0.1)
