import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import stockstats
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
import datetime
import os
from Model import Trade_Model
from dateutil.relativedelta import relativedelta
import keras.backend as K
import tensorflow as tf

global transaction_cost = 0.0005

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
    costs = tf.concat([tf.zeros((1,)), transaction_cost * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs

    sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns) * np.sqrt(252)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return -sharpe


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
    costs = tf.concat([tf.zeros((1,)), transaction_cost * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1)], axis=0)
    portfolio_returns = portfolio_returns - costs
    volat = K.std(portfolio_returns) * np.sqrt(252)

    # since we want to maximize Sharpe, while gradient descent minimizes the loss,
    #   we can negate Sharpe (the min of a negated function is its max)
    return volat - K.min([K.mean(portfolio_returns), 0])


class Simulation:
    def __init__(self, start, end, path, assets, batch_size, loss, rebalance=1):
        self.start = start
        self.end = end
        self.path = path
        self.rebalance = rebalance
        self.balance = 10000
        self.score_history = [self.balance]
        self.benchmark_history = [self.balance]
        self.daily_returns = []
        self.composition = []
        self.assets = assets
        self.batch_size = batch_size
        self.loss = loss

    def trade(self, model, data, price_data, start_date, end_date, path, asset_names=None):
        if len(self.composition) == 0:
            self.composition.append(np.zeros(shape=model.n_assets))

        benchmark_composition = self.benchmark_history[-1] / model.n_assets / price_data[0] / (1 + 0.0005)
        costs = [0]

        index = yf.download(['^GSPC'], start=start_date, end=end_date)[['Close']].to_numpy().flatten()
        index = index / index[0] * 10000
        actions = model.model.predict(data)

        for t in range(price_data.shape[0]):
            action = actions[t]
            value = self.balance + np.dot(self.composition[-1], price_data[t])
            self.composition.append(value * action / price_data[t])
            cost = 0.0005 * np.sum(price_data[t] * np.abs(self.composition[-1] - self.composition[-2]))


            self.balance = value - np.dot(self.composition[-1], price_data[t]) - cost
            self.score_history.append(value - cost)
            self.benchmark_history.append(np.dot(benchmark_composition, price_data[t]))
            costs.append(cost + costs[-1])
            self.daily_returns.append(self.score_history[-1] / self.score_history[-2] - 1)
            if self.daily_returns[-1] > 1:
                print('BUG!')
        self.balance = self.score_history[-1]
        self.composition.append(np.zeros(shape=model.n_assets))
        daily_returns = np.array(self.daily_returns)
        score_history = np.array(self.score_history)
        benchmark_history = np.array(self.benchmark_history)
        benchmark_daily_returns = benchmark_history[1:] / benchmark_history[:-1] - 1

        max_drawdown = np.max([np.max(1 - x / score_history[:T + 1]) for T, x in enumerate(score_history)])
        max_drawdown_bench = np.max([np.max(1 - x / benchmark_history[:T + 1]) for T, x in enumerate(benchmark_history)])

        #colors = np.random.uniform(0, 1, (29, 3))
        plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.tab20c.colors)
        plt.plot(np.array(actions))
        plt.legend(asset_names, fontsize='xx-small')
        #plt.show(block=True)
        plt.savefig(path + 'composition')
        plt.clf()

        plt.plot(self.score_history)

        plt.plot(index)
        plt.plot(self.benchmark_history)

        plt.legend(('Portfolio', 'Index', 'EWP'))
        #plt.show(block=True)
        plt.savefig(path + 'equity')
        plt.clf()

        plt.plot(costs)
        plt.title('Cumulative transaction cost')
        #plt.show(block=True)
        plt.savefig(path + 'transaction cost')
        plt.clf()

        excess_returns = daily_returns
        volatility = np.std(excess_returns)
        ror = (self.score_history[-1] / 10000) ** (252 / len(self.score_history)) - 1
        ror_bench = (self.benchmark_history[-1] / 10000) ** (252 / len(self.benchmark_history)) - 1
        sortino_volatility = np.std(np.minimum(excess_returns, 0))

        with open(path + 'result.txt', 'a') as f:
            f.write('Strategy:\n')
            f.write('Sharpe Ratio: {}\n'.format(np.mean(excess_returns) / volatility * np.sqrt(252)))
            f.write('Sortino Ratio: {}\n'.format(np.mean(excess_returns) / sortino_volatility * np.sqrt(252)))
            f.write('ROR: {}\n'.format(ror))
            f.write('Max Drawdown: {}\n'.format(max_drawdown))
            f.write('Volatility: {}\n'.format(volatility * np.sqrt(252)))
            f.write('Information ratio**: {}\n'.format(ror ** 2 * np.sign(ror) / (volatility * np.sqrt(252)) / max_drawdown))

            f.write('\nEWP:\n')
            f.write('Sharpe Ratio: {}\n'.format(np.mean(benchmark_daily_returns) / np.std(benchmark_daily_returns) * np.sqrt(252)))
            f.write('Sortino Ratio: {}\n'.format(np.mean(benchmark_daily_returns) / np.std(np.minimum(benchmark_daily_returns, 0)) * np.sqrt(252)))
            f.write('ROR: {}\n'.format(ror_bench))
            f.write('Max Drawdown: {}\n'.format(max_drawdown_bench))
            f.write('Volatility: {}\n'.format(np.std(benchmark_daily_returns) * np.sqrt(252)))
            f.write('Information ratio**: {}\n'.format(ror_bench ** 2 * np.sign(ror_bench) / (np.std(benchmark_daily_returns) * np.sqrt(252)) / max_drawdown_bench))

    def simulate(self):
        for year in range(self.start, self.end):
            for r in range(self.rebalance):
                month = 12 * r // self.rebalance + 1
                if month >= 10:
                    month = str(month)
                else:
                    month = '0' + str(month)
                trade_start = '{}-{}-01'.format(year, month)
                os.makedirs(self.path + '{}-{}'.format(year, month))
                os.makedirs(self.path + 'TRAINING/{}-{}'.format(year, month))
                cols_per_stock = 50

                train_full, train_price_full, validation_full, validation_price_full, stock_list = \
                    download_and_process_data(train_start=pd.to_datetime(trade_start) - relativedelta(years=5),
                                              train_end=pd.to_datetime(trade_start),
                                              validation_start=pd.to_datetime(trade_start),
                                              validation_end=pd.to_datetime(trade_start) + relativedelta(months=12//self.rebalance), cols_per_stock=cols_per_stock,
                                              n_comp=len(self.assets),
                                              download_start_shift=100, stock_lists=[self.assets],
                                              path=self.path + '{}-{}/'.format(year, month))
                model = Trade_Model(n_assets=len(stock_list), loss=self.loss, batch_size=self.batch_size)
                model.get_allocations(train_full, train_price_full)
                model.model.save(self.path + '{}-{}/model.keras'.format(year, month))

                #self.trade(model, validation_full, validation_price_full, pd.to_datetime('{}-01-01'.format(self.start)),
                #           pd.to_datetime(trade_start) + relativedelta(months=12//self.rebalance),
                #           path=self.path + '{}-{}/'.format(year, month), asset_names=stock_list)


def find_stocks(trade_start):
    check_start = pd.to_datetime(trade_start) - relativedelta(years=3)
    sp_table = pd.read_csv('S&P500 components.csv')
    stock_list = sp_table[sp_table.date <= trade_start]['tickers'].iloc[-1].split(',')
    data = yf.download(stock_list, start=check_start, end=trade_start, period='1d').dropna(axis=1)

    cols = []
    stock_list_f = []
    for stock in stock_list:
        if stock in data['Adj Close'].columns:
            cols.append(('Adj Close', stock))
            stock_list_f.append(stock)
        elif stock in data['Close'].columns:
            cols.append(('Close', stock))
            stock_list_f.append(stock)
    stock_list_f = np.array(stock_list_f)
    data = data[cols].to_numpy()

    #mask = data[-1, :] > 0.9 * np.max(data, axis=0)
    #data = data[:, mask]
    window = data.shape[0]

    returns = data[1:, :] / data[:-1, :] - 1
    ranking_short_h = np.argsort(-np.mean(returns[-window//12:], axis=0) / np.std(returns[-window//12:], axis=0))
    ranking_short_l = np.argsort(np.mean(returns[-window // 12:], axis=0) / np.std(returns[-window // 12:], axis=0))
    ranking_med_h = np.argsort(-np.mean(returns[-window//3:], axis=0) / np.std(returns[-window//3:], axis=0))
    ranking_med_l = np.argsort(np.mean(returns[-window // 3:], axis=0) / np.std(returns[-window // 3:], axis=0))
    ranking_long_h = np.argsort(-np.mean(returns, axis=0) / np.std(returns, axis=0))
    ranking_long_l = np.argsort(np.mean(returns, axis=0) / np.std(returns, axis=0))

    return stock_list_f[ranking_short_h], stock_list_f[ranking_med_h], stock_list_f[ranking_long_h], \
           stock_list_f[ranking_short_l], stock_list_f[ranking_med_l], stock_list_f[ranking_long_l]


def portfolio_value_calc(actions, price_data, start_balance=10000, t_cost=0.0005, start_composition=None, start_value=None, return_all=False):
    balance = start_balance
    if start_composition is None:
        composition = [np.zeros(price_data.shape[1])]
    else:
        composition = start_composition

    if start_value is None:
        portfolio_value = []
    else:
        portfolio_value = start_value

    for t in range(price_data.shape[0]):
        action = actions[t]
        value = balance + np.dot(composition[-1], price_data[t])
        composition.append(value * action / price_data[t])
        cost = t_cost * np.sum(price_data[t] * np.abs(composition[-1] - composition[-2]))

        balance = value - np.dot(composition[-1], price_data[t]) - cost
        portfolio_value.append(value - cost)
    if return_all is False:
        return portfolio_value
    else:
        return portfolio_value, composition, balance


def portfolio_performance(portfolio_value, save_path):
    daily_returns = portfolio_value[1:] / portfolio_value[:-1] - 1
    max_drawdown = np.max([np.max(1 - x / portfolio_value[:T[0] + 1]) for T, x in np.ndenumerate(portfolio_value)])
    arc = (portfolio_value[-1] / portfolio_value[0]) ** (252 / len(daily_returns)) - 1
    volatility = np.std(daily_returns)
    total_return = portfolio_value[-1] / portfolio_value[0] - 1
    downside_devation = np.std(np.ma.masked_array(daily_returns, daily_returns >= 0))
    with open(save_path, 'w') as f:
        f.write('Sharpe Ratio: {}\n'.format(np.mean(daily_returns) / volatility * np.sqrt(252)))
        f.write('Sortino Ratio: {}\n'.format(np.mean(daily_returns) / downside_devation * np.sqrt(252)))
        f.write('ARC: {}\n'.format(arc))
        f.write('Downside Deviation: {}\n'.format(downside_devation))
        f.write('Max Drawdown: {}\n'.format(max_drawdown))
        f.write('Volatility: {}\n'.format(volatility * np.sqrt(252)))
        f.write('Total return: {}\n'.format(total_return))
        f.write('Information ratio*: {}\n'.format(arc / volatility / np.sqrt(252)))
        f.write('Information ratio**: {}\n'.format(arc ** 2 * np.sign(arc) / (volatility * np.sqrt(252)) / max_drawdown))


def download_and_process_data(train_start, train_end, validation_start, validation_end, cols_per_stock, n_comp=None, download_start_shift=0, stock_lists=None, path='', return_dates = False):
    if stock_lists is None:
        sp_table = pd.read_csv('S&P500 components.csv')
        stock_lists = [sp_table[pd.to_datetime(sp_table.date) <= pd.to_datetime(train_end) - relativedelta(years=5)][
                      'tickers'].iloc[-1].split(',')]

    #other_list = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ',
    #              'IYM', 'IYR', 'IYW', 'IDU']

    other_list = []
    download_train_start = pd.to_datetime(train_start) - datetime.timedelta(days=download_start_shift)
    download_validation_start = pd.to_datetime(validation_start) - datetime.timedelta(days=download_start_shift)
    final_stock_list = []

    train_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    validation_full = stockstats.wrap(yf.download('^GSPC', start=download_validation_start, end=validation_end))[['close']]
    train_price_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    validation_price_full = stockstats.wrap(yf.download('^GSPC', start=download_validation_start, end=validation_end))[['close']]

    if len(stock_lists) > 0:
        full_train_data = yf.download(list(stock_lists[0]), start=download_train_start, end=train_end, period='1d',
                                  group_by='ticker').dropna(axis=0)
        full_validation_data = yf.download(list(stock_lists[0]), start=download_validation_start, end=validation_end, period='1d',
                                       group_by='ticker').dropna(axis=0)

    for stock_list in stock_lists:
        stock_list = list(stock_list)
        count = 0
        for stock in stock_list:
            if stock not in final_stock_list and stock in full_train_data.columns.get_level_values(0) and stock in full_validation_data.columns.get_level_values(0):
                train_data = full_train_data[stock]
                validation_data = full_validation_data[stock]
                if 'Adj Close' in train_data.columns:
                    train_data.loc[:, 'Low'] = train_data['Low'] * train_data['Adj Close'] / train_data['Close']
                    train_data.loc[:, 'High'] = train_data['High'] * train_data['Adj Close'] / train_data['Close']
                    train_data = train_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
                if 'Adj Close' in validation_data.columns:
                    validation_data.loc[:, 'Low'] = validation_data['Low'] * validation_data['Adj Close'] / validation_data['Close']
                    validation_data.loc[:, 'High'] = validation_data['High'] * validation_data['Adj Close'] / train_data['Close']
                    validation_data = validation_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})

                col_names = ['close_1_roc'] + ['close_1_roc_-{}_s'.format(i + 1) for i in range(cols_per_stock - 1)]
                #col_names2 = ['volume_1_roc'] + ['volume_1_roc_-{}_s'.format(i + 1) for i in range(cols_per_stock//2 - 1)]
                if len(train_data > 0) and len(validation_data > 0):
                    train_stock_data = stockstats.wrap(train_data)[col_names]
                    #train_stock_data2 = stockstats.wrap(train_data)[col_names2]
                    train_price_data = stockstats.wrap(train_data)[['close']]
                    validation_stock_data = stockstats.wrap(validation_data)[col_names]
                    #validation_stock_data2 = stockstats.wrap(validation_data)[col_names2]
                    validation_price_data = stockstats.wrap(validation_data)[['close']]

                    # train_stock_data.drop(columns=['close_1_roc'], inplace=True)
                    # validation_stock_data.drop(columns=['close_1_roc'], inplace=True)

                    train_full = train_full.join(train_stock_data, rsuffix=stock, how='inner')
                    # train_full = train_full.join(train_stock_data2, rsuffix=stock)
                    train_price_full = train_price_full.join(train_price_data, rsuffix=stock, how='inner')
                    validation_full = validation_full.join(validation_stock_data, rsuffix=stock, how='inner')
                    # validation_full = validation_full.join(validation_stock_data2, rsuffix=stock)
                    validation_price_full = validation_price_full.join(validation_price_data, rsuffix=stock,
                                                                       how='inner')
                    count += 1
                    final_stock_list.append(stock)

                if n_comp is not None and (n_comp - len(other_list)) // len(stock_lists) <= count:
                    break


    train_full.drop(columns=['close'], inplace=True)
    validation_full.drop(columns=['close'], inplace=True)
    train_price_full.drop(columns=['close'], inplace=True)
    validation_price_full.drop(columns=['close'], inplace=True)
    technicals = [('^VIX', 'close_1_roc'), ('^IRX', 'close'), ('^FVX', 'close')]

    for tpl in technicals:
        symbol, feat = tpl
        train_data = yf.download(symbol, start=download_train_start, end=train_end).dropna(axis=0)
        validation_data = yf.download(symbol, start=download_validation_start, end=validation_end).dropna(axis=0)
        col_names = [feat] + ['{}_-{}_s'.format(feat, i + 1) for i in range(cols_per_stock - 1)]
        train_stock_data = stockstats.wrap(train_data)[col_names]
        validation_stock_data = stockstats.wrap(validation_data)[col_names]
        train_full = train_full.join(train_stock_data, rsuffix=symbol, how='inner')
        validation_full = validation_full.join(validation_stock_data, rsuffix=symbol, how='inner')

    dates = train_full.loc[train_start:].index
    train_full = train_full.loc[train_start:].to_numpy() / 100
    #train_full[:, :-len(col_names)] = train_full[:, :-len(col_names)] / 100
    train_price_full = train_price_full[train_start:].to_numpy()
    validation_full = validation_full[validation_start:].to_numpy() / 100
    #validation_full[:, :-len(col_names)] = validation_full[:, :-len(col_names)] / 100
    validation_price_full = validation_price_full[validation_start:].to_numpy()

    features = len(final_stock_list) + len(technicals)
    train_final = np.zeros(shape=(train_full.shape[0], train_full.shape[1] // features, features))
    validation_final = np.zeros(shape=(validation_full.shape[0], validation_full.shape[1] // features, features))

    #np.savetxt(path + 'current_train.txt', train_full)
    #np.savetxt(path + 'current_train_price.txt', train_price_full)
    #np.savetxt(path + 'current_validation.txt', validation_full)
    #np.savetxt(path + 'current_validation_price.txt', validation_price_full)
    np.savetxt(path + 'current stock_list.txt', final_stock_list, fmt='%s')

    for i in range(features):
        train_final[:, :, i] = train_full[:, i * cols_per_stock: (i+1) * cols_per_stock]
        validation_final[:, :, i] = validation_full[:, i * cols_per_stock: (i + 1) * cols_per_stock]

    if not return_dates:
        return train_final, train_price_full, validation_final, validation_price_full, final_stock_list
    else:
        return train_final, train_price_full, validation_final, validation_price_full, final_stock_list, dates



def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):

    def _init() -> gym.Env:
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def data_load(path=''):
    train_full = np.loadtxt(path+'current_train.txt')
    train_price = np.loadtxt(path+'current_train_price.txt')
    validation_full = np.loadtxt(path+'current_validation.txt')
    validation_price = np.loadtxt(path+'current_validation_price.txt')
    with open(path+'current stock_list.txt', 'r') as f:
        final_stock_list = f.read().split('\n')[:-1]

    return train_full, train_price, validation_full, validation_price, final_stock_list


# param_dict = {'gamma':[0.9, 0.99], 'n_steps':[64, 128], 'learning_rate': [0.001, 0.0015, 0.002]}
# params_to_test = grid_search(param_dict)


if __name__ == '__main__':
    assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA',
              'IYJ',
              'IYM', 'IYR', 'IYW', 'IDU']
    assets_old = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH',
              'IYJ',
              'IYM', 'IYR', 'IYW', 'IDU']
    assets_no_bonds = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'IYJ',
                       'IYM', 'IYR', 'IYW', 'IDU']
    assets_diverse = ['IVV', 'IJH', 'IJR', 'IYK', 'IYC', 'IYH', 'IYW', 'IDU', 'GC=F', 'CL=F', 'HG=F', 'JPY=X', 'CHF=X',
                      'AUD=X', 'IEF']
    assets_div_limited = ['IVV', 'IJH', 'IJR', 'IEF', 'GC=F', 'CL=F', 'HG=F', 'JPY=X', 'CHF=X', 'AUD=X']
    assets_full_diverse = ['IEF', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU',
                           'GC=F', 'CL=F', 'HG=F',
                           'ZC=F', 'KC=F']

    dow_jones = ['MMM', 'AXP', 'T', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DD', 'XOM',
                 'GE', 'GS', 'HD', 'INTC', 'IBM', 'JPM', 'JNJ', 'MCD', 'MRK', 'MSFT',
                 'NKE', 'PFE', 'PG', 'TRV', 'RTX', 'UNH', 'VZ', 'V', 'WMT', 'DIS']

    global transaction_cost = 0
    simulation = Simulation(2008, 2024, 'p1 sharpe 64 c0/', loss=sharpe_loss, batch_size=64, assets=assets_old, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p1 vol 64 c0/', loss=soft_target_simple, batch_size=64, assets=assets_old, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p3 sharpe 64 c0/', loss=sharpe_loss, batch_size=64, assets=assets_full_diverse, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p3 vol 64 c0/', loss=soft_target_simple, batch_size=64, assets=assets_full_diverse, rebalance=4)
    simulation.simulate()

    global transaction_cost = 0.1
    simulation = Simulation(2008, 2024, 'p1 sharpe 64 c1/', loss=sharpe_loss, batch_size=64, assets=assets_old, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p1 vol 64 c1/', loss=soft_target_simple, batch_size=64, assets=assets_old, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p3 sharpe 64 c1/', loss=sharpe_loss, batch_size=64, assets=assets_full_diverse, rebalance=4)
    simulation.simulate()

    simulation = Simulation(2008, 2024, 'p3 vol 64 c1/', loss=soft_target_simple, batch_size=64, assets=assets_full_diverse, rebalance=4)
    simulation.simulate()
