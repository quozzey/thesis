import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import yfinance as yf
from main_final import portfolio_performance, portfolio_value_calc, download_and_process_data, load_data, statistic
from dateutil.relativedelta import relativedelta
import keras.backend as K
from scipy.optimize import minimize

#data = yf.download(['^VIX', '^IRX', '^FVX'], start='2008-06-01', end='2023-12-31', group_by='column')
#data.to_csv('indicators.csv')


# Activation function to neural network (generating weights)
def softmax_mod(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1) / tf.expand_dims(
        tf.reduce_sum(tf.exp(tf.abs(x)) - 1, axis=1), axis=-1)

# Function to minimise in Minimum Volatility approach
def soft_target_simple(returns, y_pred):
    portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs
    volat = K.std(portfolio_returns) * np.sqrt(252)
    ret = (K.exp(tf.reduce_mean(K.log(1 + portfolio_returns), axis=0)) ** 252 - 1)

    return volat - K.min([ret, 0])


# Data for VIX vs VXX plot in Chapter 1

#comp = yf.download(['^VIX', 'VXX'], start='2018-02-01', end='2023-12-31')['Adj Close']
#comp.iloc[:, 0] = comp.iloc[:, 0] / comp.iloc[0, 0]
#comp.iloc[:, 1] = comp.iloc[:, 1] / comp.iloc[0, 1]

#VIX vs VXX plot

#ax[0].grid(ls='--', which='both')
#ax[0].plot(comp['VXX'])
#ax[0].title.set_text('VXX')
#ax[1].grid(ls='--', which='both')
#ax[1].plot(comp['^VIX'], c='orange')
#ax[1].title.set_text('^VIX')
#plt.grid(ls='--')
#plt.yscale('log')
#plt.plot(comp)

#plt.legend(['VXX', 'VIX'])
#plt.show()


# Calculation of portfolio weights in mean-variance benchmark
def calculate_mean_variance_weights(returns):
    T, N = returns.shape
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)

    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        res = np.sum(weights * mean_returns) / np.sqrt(portfolio_variance)
        return -res
    # Constraints for long + short portfolio
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(np.abs(weights)) - 1},)

    # Constraints for long-only portfolio
    #constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
    #               {'type': 'ineq', 'fun': lambda weights: np.min(weights)})

    initial_weights = np.ones(N) / N
    result = minimize(objective, initial_weights, constraints=constraints, tol=0.0001)

    optimal_weights = result.x

    return optimal_weights


def sharpe_loss(returns, y_pred):
    portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs

    sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns) * np.sqrt(252)

    return -sharpe


def month(r, rebalances):
    mon = 12 * r // rebalances + 1
    if mon >= 10:
        mon = str(mon)
    else:
        mon = '0' + str(mon)
    return mon


assets1 = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ',
                  'IYM', 'IYR', 'IYW', 'IDU']

assets2 = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ',
                  'IYM', 'IYR', 'IYW', 'IDU']

assets3 = ['IEF', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU',
                           'GC=F', 'CL=F', 'HG=F',
                           'ZC=F', 'KC=F']

all_assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM',
              'IYR', 'IYW', 'IDU', 'GC=F', 'CL=F', 'HG=F', 'ZC=F', 'KC=F']

start = 2014
end = 2024
rebalances = 4

model_data, price_data_learn, _, _, _, learn_dates = load_data(train_start=pd.to_datetime('2013-01-01'),
                                              train_end=pd.to_datetime('2024-01-01'),
                                              test_start=pd.to_datetime('2023-12-31'),
                                              test_end=pd.to_datetime('2024-01-01'), cols_per_stock=50,
                                              n_comp=len(assets3),
                                              download_start_shift=100, stock_lists=[assets3],
                                              path='dump/', return_dates=True)

save_path = 'RESULTS/'


dates = learn_dates[learn_dates >= pd.to_datetime('2014-01-01')]
price_data = price_data_learn[learn_dates >= pd.to_datetime('2014-01-01')]
model_data = model_data[learn_dates >= pd.to_datetime('2014-01-01')]
ewp_portfolio = np.ones(shape=price_data.shape) / len(assets1)
ewp = np.array(portfolio_value_calc(ewp_portfolio, price_data))

sp500 = yf.download(['^GSPC'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
sp500 = sp500 / sp500[0] * 10000

dji = yf.download(['^DJI'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
dji = dji / dji[0] * 10000

portfolio_performance(ewp, '{}/ewp.txt'.format(save_path))
portfolio_performance(sp500, '{}/sp500.txt'.format(save_path))
portfolio_performance(dji, '{}/djia.txt'.format(save_path))


# MEAN-VARIANCE PORTFOLIO PERFORMANCE
portfolio_value = []
composition = None
balance = 10000
for year in range(start, end):
    for r in range(rebalances):
        learn_start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances) - relativedelta(years=1)
        start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
        end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
        price_data_train = price_data_learn[np.logical_and(learn_start_day <= learn_dates, learn_dates < start_day)]
        price_data_trade = price_data_learn[np.logical_and(start_day <= learn_dates, learn_dates < end_day)]
        actions = calculate_mean_variance_weights(price_data_train[1:] / price_data_train[:-1] - 1)
        actions = np.array([actions for _ in range(price_data_trade.shape[0])])

        portfolio_value, composition, balance = portfolio_value_calc(actions, price_data_trade, start_composition=composition, start_value=portfolio_value, start_balance=balance, return_all=True)
mean_var = np.array(portfolio_value)
portfolio_performance(mean_var, '{}/mean var.txt'.format(save_path))


# SHARPE RATIO LSTM PERFORMANCE
portfolio_value = []
composition = None
balance = 10000
for year in range(start, end):
    for r in range(rebalances):
        start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
        end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
        model_data_lim = model_data[np.logical_and(start_day <= dates, dates < end_day)]
        price_data_lim = price_data[np.logical_and(start_day <= dates, dates < end_day)]
        model = keras.saving.load_model('p3 /{}-{}/model.keras'.format(year, month(r, rebalances)),
                                        safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple, 'sharpe_loss': sharpe_loss})
        portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                            start_composition=composition, start_value=portfolio_value, start_balance=balance, return_all=True)

lstm_shr = np.array(portfolio_value)
portfolio_performance(lstm_shr, '{}/lstm-shr.txt'.format(save_path))


# MINIMUM VOLATILITY LSTM PERFORMANCE
portfolio_value = []
composition = None
balance = 10000
for year in range(start, end):
    for r in range(rebalances):
        start_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * r // rebalances)
        end_day = pd.to_datetime('{}-01-01'.format(year)) + relativedelta(months=12 * (r + 1) // rebalances)
        model_data_lim = model_data[np.logical_and(start_day <= dates, dates < end_day)]
        price_data_lim = price_data[np.logical_and(start_day <= dates, dates < end_day)]
        model = keras.saving.load_model('vol long only1/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                        safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple, 'softmax_mod': softmax_mod})
        portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                            start_composition=composition, start_value=portfolio_value, start_balance=balance, return_all=True)
lstm_vol = np.array(portfolio_value)
portfolio_performance(lstm_vol, '{}/lstm-vol.txt'.format(save_path))


plt.plot(dates, sp500)
plt.plot(dates, dji)
plt.plot(dates, ewp)
plt.plot(dates, mean_var)
plt.plot(dates, lstm_shr)
plt.plot(dates, lstm_vol)
plt.yscale('log')
plt.legend(['S&P500', 'DJIA', 'EWP', 'Mean-Var', 'LSTM-ShR', 'LSTM-Vol'])
plt.grid(ls='--', which='both')

plt.show()
