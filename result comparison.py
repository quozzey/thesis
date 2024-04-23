import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import yfinance as yf
from main_final import portfolio_performance, portfolio_value_calc, download_and_process_data
from dateutil.relativedelta import relativedelta
import keras.backend as K
from scipy.optimize import minimize
import seaborn as sns




def min_var(returns, y_pred):
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
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs

    volat = K.std(portfolio_returns) * np.sqrt(252)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return volat


def soft_target_simple(returns, y_pred):
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
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs
    volat = K.std(portfolio_returns) * np.sqrt(252)
    ret = (K.exp(tf.reduce_mean(K.log(1 + portfolio_returns), axis=0)) ** 252 - 1)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return volat - K.min([ret, 0])

#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

#ax[0].grid(ls='--', which='both')
#ax[0].plot(comp['VXX'])
#ax[0].title.set_text('VXX')
#ax[1].grid(ls='--', which='both')
#ax[1].plot(comp['^VIX'], c='orange')
#ax[1].title.set_text('^VIX')


#plt.plot(comp)
#plt.legend(['VXX', 'VIX'])
#plt.show()


def calculate_mean_variance_weights(returns):

    # Number of assets (N) and number of days (T)
    T, N = returns.shape

    # Calculate mean returns and covariance matrix
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)

    # Define objective function to minimize (portfolio variance)
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        res = np.sum(weights * mean_returns) / np.sqrt(portfolio_variance)
        return -res

    # Define constraint (sum of weights equals 1)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(np.abs(weights)) - 1})

    # Define initial guess for weights
    initial_weights = np.ones(N) / N

    # Minimize the objective function subject to the constraint
    result = minimize(objective, initial_weights, constraints=constraints, tol=0.0001)

    if result.success:
        optimal_weights = result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

    return optimal_weights


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
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs

    sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns) * np.sqrt(252)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return -sharpe


def soft_target(returns, y_pred):
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
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs
    volat = K.std(portfolio_returns) * np.sqrt(252)

    res = K.mean(portfolio_returns) * tf.exp(-0.25 * (100 * volat - 5) ** 2)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return -res

def month(r, rebalances):
    mon = 12 * r // rebalances + 1
    if mon >= 10:
        mon = str(mon)
    else:
        mon = '0' + str(mon)
    return mon


assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'IYJ',
                  'IYM', 'IYR', 'IYW', 'IDU']
assets_no_bonds = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ',
                  'IYM', 'IYR', 'IYW', 'IDU']
assets_commods = ['TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU', 'GC=F', 'HG=F', 'CL=F', 'ZC=F', 'NG=F']
#assets_diverse = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU', 'GC=F', 'CL=F', 'ZC=F', 'EUR=X', 'JPY=X', 'CHF=X',
#                  'SI=F', 'GBP=X', 'CC=F', 'AUD=X', 'PL=F', 'HG=F', 'TLT', 'IEF', 'SHY']
assets_diverse = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM',
                                  'IYR', 'IYW', 'IDU', 'GC=F', 'CL=F', 'ZC=F', 'EUR=X', 'JPY=X', 'CHF=X',
                                  'GBP=X', 'AUD=X', 'PL=F', 'IEF']
assets_div_limited = ['IVV', 'IJH', 'IJR', 'IEF', 'GC=F', 'CL=F', 'HG=F', 'JPY=X', 'CHF=X', 'AUD=X']

all_assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM',
              'IYR', 'IYW', 'IDU', 'GC=F', 'CL=F', 'HG=F', 'ZC=F', 'KC=F']

assets_full_diverse = ['IEF', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU',
                           'GC=F', 'CL=F', 'HG=F',
                           'ZC=F', 'KC=F']

#comp = pd.read_csv('DATA.csv', header=[0, 1], index_col=0)




start = 2014
end = 2024
rebalances = 4

model_data, price_data_learn, _, _, _, learn_dates = download_and_process_data(train_start=pd.to_datetime('2013-01-01'),
                                              train_end='2024-01-01',
                                              validation_start='2024-01-01',
                                              validation_end='2024-01-02', cols_per_stock=50,
                                              n_comp=len(assets),
                                              download_start_shift=100, stock_lists=[assets],
                                              path='FINAL RESULTS/', return_dates=True)


#close_roc = 100 * model_data[:, 0, :-3].reshape((model_data.shape[0], model_data.shape[2] - 3))
#covariances = np.corrcoef(close_roc, rowvar=False)
#sns.heatmap(covariances, cmap='hot', annot=True, xticklabels=all_assets, yticklabels=all_assets, fmt='.2f', vmin=-1, vmax=1)
#plt.show()


dates = learn_dates[learn_dates >= pd.to_datetime('2014-01-01')]
price_data = price_data_learn[learn_dates >= pd.to_datetime('2014-01-01')]
model_data = model_data[learn_dates >= pd.to_datetime('2014-01-01')]
ewp_portfolio = np.ones(shape=price_data.shape) / len(assets_no_bonds)
#sp500 = yf.download(['^GSPC'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
#sp500 = sp500 / sp500[0] * 10000

#djia = yf.download(['^DJI'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
#djia = djia / djia[0] * 10000

#portfolio_performance(sp500, 'PORTFOLIO NO BONDS/sp500.txt')
#portfolio_performance(djia, 'PORTFOLIO NO BONDS/djia.txt')

#ewp_portfolio_value = np.array(portfolio_value_calc(ewp_portfolio, price_data))
#portfolio_performance(ewp_portfolio_value, 'PORTFOLIO NO BONDS/ewp.txt')

t_costs = [0, 0.0002, 0.0005, 0.001]
model0 = []
model5 = []
model10 = []


for t_cost in t_costs:
    portfolio_value = []
    composition = None
    balance = 10000
    for year in range(start, end):
        for r in range(rebalances):
            start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
            end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
            model_data_lim = model_data[np.logical_and(start_day <= dates, dates < end_day)]
            price_data_lim = price_data[np.logical_and(start_day <= dates, dates < end_day)]
            model = keras.saving.load_model('0 transaction TRAIN/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'sharpe_loss': sharpe_loss})
            portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                                         start_composition=composition,
                                                                         start_value=portfolio_value,
                                                                         start_balance=balance, return_all=True,
                                                                         t_cost=t_cost)
    model0.append(np.array(portfolio_value))

    portfolio_value = []
    composition = None
    balance = 10000
    for year in range(start, end):
        for r in range(rebalances):
            start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
            end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
            model_data_lim = model_data[np.logical_and(start_day <= dates, dates < end_day)]
            price_data_lim = price_data[np.logical_and(start_day <= dates, dates < end_day)]
            model = keras.saving.load_model('ADJ CLOSE MODEL/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'sharpe_loss': sharpe_loss})
            portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                                         start_composition=composition,
                                                                         start_value=portfolio_value,
                                                                         start_balance=balance, return_all=True,
                                                                         t_cost=t_cost)
    model5.append(np.array(portfolio_value))

    portfolio_value = []
    composition = None
    balance = 10000
    for year in range(start, end):
        for r in range(rebalances):
            start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
            end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
            model_data_lim = model_data[np.logical_and(start_day <= dates, dates < end_day)]
            price_data_lim = price_data[np.logical_and(start_day <= dates, dates < end_day)]
            model = keras.saving.load_model('01 transaction TRAIN/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'sharpe_loss': sharpe_loss})
            portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                                         start_composition=composition,
                                                                         start_value=portfolio_value,
                                                                         start_balance=balance, return_all=True,
                                                                         t_cost=t_cost)
    model10.append(np.array(portfolio_value))


fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
axs.set_yscale('log')
for i, ax in enumerate(axs):
    ax.plot(dates, model0[i])
    ax.plot(dates, model5[i])
    ax.plot(dates, model10[i])




#plt.plot(dates, sp500)
#plt.plot(dates, djia)
#plt.plot(dates, ewp_portfolio_value)
#plt.plot(dates, mean_var)
#plt.plot(dates,sharpe)
#plt.plot(dates, min_vol)
#plt.yscale('log')
#plt.legend(['S&P500', 'DJIA', 'EWP', 'Mean-Var', 'LSTM-ShR', 'LSTM-Min-Vol'])
#plt.grid(ls='--', which='both')

plt.show()
