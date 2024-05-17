import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from main_final import portfolio_performance, portfolio_value_calc, download_and_process_data
from dateutil.relativedelta import relativedelta
import keras.backend as K
from scipy.optimize import minimize


def softmax_mod(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1) / tf.expand_dims(
        tf.reduce_sum(tf.exp(tf.abs(x)) - 1, axis=1), axis=-1)


def soft_target_simple(returns, y_pred):
    portfolio_returns = tf.reduce_sum(tf.multiply(returns, y_pred), axis=1)
    costs = tf.concat([tf.zeros((1,)), 0.0005 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs
    volat = K.std(portfolio_returns) * np.sqrt(252)
    ret = (K.exp(tf.reduce_mean(K.log(1 + portfolio_returns), axis=0)) ** 252 - 1)

    return volat - K.min([ret, 0])


def calculate_mean_variance_weights(returns):
    T, N = returns.shape
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)

    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        res = np.sum(weights * mean_returns) / np.sqrt(portfolio_variance)
        return -res

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(np.abs(weights)) - 1})
    initial_weights = np.ones(N) / N

    result = minimize(objective, initial_weights, constraints=constraints, tol=0.0001)

    if result.success:
        optimal_weights = result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

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


all_assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM',
              'IYR', 'IYW', 'IDU', 'GC=F', 'CL=F', 'HG=F', 'ZC=F', 'KC=F']

assets3 = ['IEF', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU',
                       'GC=F', 'CL=F', 'HG=F',
                       'ZC=F', 'KC=F']

start = 2014
end = 2024
rebalances = 4

model_data, price_data_learn, _, _, _, learn_dates = download_and_process_data(train_start=pd.to_datetime('2014-01-01'),
                                                                               train_end='2024-01-01',
                                                                               test_start='2024-01-01',
                                                                               test_end='2024-01-02',
                                                                               cols_per_stock=50,
                                                                               n_comp=len(all_assets),
                                                                               download_start_shift=100,
                                                                               stock_lists=[all_assets],
                                                                               path='FINAL RESULTS/', return_dates=True)

#close_roc = 100 * model_data[:, 0, :-3].reshape((model_data.shape[0], model_data.shape[2] - 3))
#covariances = np.corrcoef(close_roc, rowvar=False)
#sns.heatmap(covariances, cmap='RdYlGn', annot=True, xticklabels=all_assets, yticklabels=all_assets, fmt='.2f', vmin=-1, vmax=1)
#plt.show()


dates = learn_dates[learn_dates >= pd.to_datetime('2014-01-01')]
price_data = price_data_learn[learn_dates >= pd.to_datetime('2014-01-01')]
model_data = model_data[learn_dates >= pd.to_datetime('2014-01-01')]
ewp_portfolio = np.ones(shape=price_data.shape) / len(assets2)
# sp500 = yf.download(['^GSPC'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
# sp500 = sp500 / sp500[0] * 10000

# djia = yf.download(['^DJI'], start='2014-01-01', end='2024-01-01').loc[dates, 'Adj Close'].to_numpy().flatten()
# djia = djia / djia[0] * 10000

# portfolio_performance(sp500, 'PORTFOLIO NO BONDS/sp500.txt')
# portfolio_performance(djia, 'PORTFOLIO NO BONDS/djia.txt')

# ewp_portfolio_value = np.array(portfolio_value_calc(ewp_portfolio, price_data))
# portfolio_performance(ewp_portfolio_value, 'PORTFOLIO NO BONDS/ewp.txt')

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
            model = keras.saving.load_model('vol 0 corr/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple,
                                                                             'softmax_mod': softmax_mod})
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
            model = keras.saving.load_model('vol 64/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple,
                                                                             'softmax_mod': softmax_mod})
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
            model = keras.saving.load_model('vol 01 corr/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                            safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple,
                                                                             'softmax_mod': softmax_mod})
            portfolio_value, composition, balance = portfolio_value_calc(model.predict(model_data_lim), price_data_lim,
                                                                         start_composition=composition,
                                                                         start_value=portfolio_value,
                                                                         start_balance=balance, return_all=True,
                                                                         t_cost=t_cost)
    model10.append(np.array(portfolio_value))

models = [model0, model5, model10]
for j, model in enumerate(models):
    for i, t_cost in enumerate(t_costs):
        portfolio_performance(model[i], 'volatility costs/model {}-{}.txt'.format(5 * j, t_cost))

# plt.plot(dates, sp500)
# plt.plot(dates, djia)
# plt.plot(dates, ewp_portfolio_value)
# plt.plot(dates, mean_var)
# plt.plot(dates,sharpe)
# plt.plot(dates, min_vol)
# plt.yscale('log')
# plt.legend(['S&P500', 'DJIA', 'EWP', 'Mean-Var', 'LSTM-ShR', 'LSTM-Min-Vol'])
# plt.grid(ls='--', which='both')

#plt.show()
