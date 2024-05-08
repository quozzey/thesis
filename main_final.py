import numpy as np
import pandas as pd
import yfinance as yf
import stockstats
import datetime
import os
from Model import Trade_Model
from dateutil.relativedelta import relativedelta


class Simulation:
    def __init__(self, start, end, path, assets, batch_size, rebalance=1, t_cost=0.0005):
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
        self.t_cost = t_cost

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

                train_full, train_price_full, test_full, test_price_full, stock_list = \
                    download_and_process_data(train_start=pd.to_datetime('2008-06-01'),
                                              train_end=pd.to_datetime(trade_start),
                                              test_start=pd.to_datetime(trade_start),
                                              test_end=pd.to_datetime(trade_start) + relativedelta(months=12//self.rebalance), cols_per_stock=cols_per_stock,
                                              n_comp=len(self.assets),
                                              download_start_shift=100, stock_lists=[self.assets],
                                              path=self.path + '{}-{}/'.format(year, month))
                model = Trade_Model(n_assets=len(stock_list), batch_size=self.batch_size, t_cost=self.t_cost)
                model.get_allocations(train_full, train_price_full)
                model.model.save(self.path + '{}-{}/model.keras'.format(year, month))


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
        f.write('Total return: {n:.2f}%\n'.format(n=100*total_return))
        f.write('ARC: {n:.2f}%\n'.format(n=100*arc))
        f.write('Volatility: {n:.2f}%\n'.format(n=100*volatility * np.sqrt(252)))
        f.write('Downside Deviation: {n:.2f}%\n'.format(n=100*downside_devation * np.sqrt(252)))
        f.write('Max Drawdown: {n:.2f}%\n'.format(n=100*max_drawdown))
        f.write('Sharpe Ratio: {n:.3f}\n'.format(n=np.mean(daily_returns) / volatility * np.sqrt(252)))
        f.write('Sortino Ratio: {n:.3f}\n'.format(n=np.mean(daily_returns) / downside_devation * np.sqrt(252)))
        f.write('Information ratio*: {n:.3f}\n'.format(n=arc / volatility / np.sqrt(252)))
        f.write('Information ratio**: {n:.3f}\n'.format(n=arc ** 2 * np.sign(arc) / (volatility * np.sqrt(252)) / max_drawdown))


def download_and_process_data(train_start, train_end, test_start, test_end, cols_per_stock, n_comp=None, download_start_shift=0, stock_lists=None, path='', return_dates = False):
    if stock_lists is None:
        sp_table = pd.read_csv('S&P500 components.csv')
        stock_lists = [sp_table[pd.to_datetime(sp_table.date) <= pd.to_datetime(train_end) - relativedelta(years=5)][
                      'tickers'].iloc[-1].split(',')]
    download_train_start = pd.to_datetime(train_start) - datetime.timedelta(days=download_start_shift)
    download_test_start = pd.to_datetime(test_start) - datetime.timedelta(days=download_start_shift)
    final_stock_list = []

    train_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    test_full = stockstats.wrap(yf.download('^GSPC', start=download_test_start, end=test_end))[['close']]
    train_price_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    test_price_full = stockstats.wrap(yf.download('^GSPC', start=download_test_start, end=test_end))[['close']]

    if len(stock_lists) > 0:
        full_train_data = yf.download(list(stock_lists[0]), start=download_train_start, end=train_end, period='1d',
                                  group_by='ticker').dropna(axis=0)
        full_test_data = yf.download(list(stock_lists[0]), start=download_test_start, end=test_end, period='1d',
                                       group_by='ticker').dropna(axis=0)

    for stock_list in stock_lists:
        stock_list = list(stock_list)
        count = 0
        for stock in stock_list:
            if stock not in final_stock_list and stock in full_train_data.columns.get_level_values(0) and stock in full_test_data.columns.get_level_values(0):
                train_data = full_train_data[stock]
                test_data = full_test_data[stock]
                if 'Adj Close' in train_data.columns:
                    train_data.loc[:, 'Low'] = train_data['Low'] * train_data['Adj Close'] / train_data['Close']
                    train_data.loc[:, 'High'] = train_data['High'] * train_data['Adj Close'] / train_data['Close']
                    train_data = train_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
                if 'Adj Close' in test_data.columns:
                    test_data.loc[:, 'Low'] = test_data['Low'] * test_data['Adj Close'] / test_data['Close']
                    test_data.loc[:, 'High'] = test_data['High'] * test_data['Adj Close'] / train_data['Close']
                    test_data = test_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})

                col_names = ['close_1_roc'] + ['close_1_roc_-{}_s'.format(i + 1) for i in range(cols_per_stock - 1)]
                if len(train_data > 0) and len(test_data > 0):
                    train_wrapped_data = stockstats.wrap(train_data)[col_names]
                    train_price_data = stockstats.wrap(train_data)[['close']]
                    test_wrapped_data = stockstats.wrap(test_data)[col_names]
                    test_price_data = stockstats.wrap(test_data)[['close']]
                    train_full = train_full.join(train_wrapped_data, rsuffix=stock, how='inner')
                    train_price_full = train_price_full.join(train_price_data, rsuffix=stock, how='inner')
                    test_full = test_full.join(test_wrapped_data, rsuffix=stock, how='inner')
                    test_price_full = test_price_full.join(test_price_data, rsuffix=stock,
                                                                       how='inner')
                    count += 1
                    final_stock_list.append(stock)
                if n_comp is not None and n_comp // len(stock_lists) <= count:
                    break

    train_full.drop(columns=['close'], inplace=True)
    test_full.drop(columns=['close'], inplace=True)
    train_price_full.drop(columns=['close'], inplace=True)
    test_price_full.drop(columns=['close'], inplace=True)
    technicals = [('^VIX', 'close_1_roc'), ('^IRX', 'close'), ('^FVX', 'close')]

    for tpl in technicals:
        symbol, feat = tpl
        train_data = yf.download(symbol, start=download_train_start, end=train_end).dropna(axis=0)
        test_data = yf.download(symbol, start=download_test_start, end=test_end).dropna(axis=0)
        col_names = [feat] + ['{}_-{}_s'.format(feat, i + 1) for i in range(cols_per_stock - 1)]
        train_wrapped_data = stockstats.wrap(train_data)[col_names]
        test_wrapped_data = stockstats.wrap(test_data)[col_names]
        train_full = train_full.join(train_wrapped_data, rsuffix=symbol)
        test_full = test_full.join(test_wrapped_data, rsuffix=symbol)

    dates = train_full.ffill().loc[train_start:].index
    train_full = train_full.ffill().loc[train_start:].to_numpy() / 100
    train_price_full = train_price_full.ffill()[train_start:].to_numpy()
    test_full = test_full.ffill().loc[test_start:].to_numpy() / 100
    test_price_full = test_price_full.ffill()[test_start:].to_numpy()

    features = len(final_stock_list) + len(technicals)
    train_final = np.zeros(shape=(train_full.shape[0], train_full.shape[1] // features, features))
    test_final = np.zeros(shape=(test_full.shape[0], test_full.shape[1] // features, features))

    for i in range(features):
        train_final[:, :, i] = train_full[:, i * cols_per_stock: (i+1) * cols_per_stock]
        test_final[:, :, i] = test_full[:, i * cols_per_stock: (i + 1) * cols_per_stock]

    if not return_dates:
        return train_final, train_price_full, test_final, test_price_full, final_stock_list
    else:
        return train_final, train_price_full, test_final, test_price_full, final_stock_list, dates


def load_data(train_start, train_end, test_start, test_end, cols_per_stock, n_comp=None, download_start_shift=0, stock_lists=None, path='', return_dates=False):
    data = pd.read_csv('DATA.csv', header=[0, 1], index_col=0)
    data.index = pd.to_datetime(data.index)
    download_train_start = pd.to_datetime(train_start) - datetime.timedelta(days=download_start_shift)
    download_test_start = pd.to_datetime(test_start) - datetime.timedelta(days=download_start_shift)
    final_stock_list = []

    train_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    test_full = stockstats.wrap(yf.download('^GSPC', start=download_test_start, end=test_end))[['close']]
    train_price_full = stockstats.wrap(yf.download('^GSPC', start=download_train_start, end=train_end))[['close']]
    test_price_full = stockstats.wrap(yf.download('^GSPC', start=download_test_start, end=test_end))[['close']]

    for stock_list in stock_lists:
        stock_list = list(stock_list)
        count = 0
        for stock in stock_list:
            train_data = data[download_train_start:train_end].xs(stock, axis=1, level=1).dropna()
            test_data = data[test_start:test_end].xs(stock, axis=1, level=1).dropna()
            train_data.loc[:, 'Low'] = train_data['Low'] * train_data['Adj Close'] / train_data['Close']
            train_data.loc[:, 'High'] = train_data['High'] * train_data['Adj Close'] / train_data['Close']
            train_data = train_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
            test_data.loc[:, 'Low'] = test_data['Low'] * test_data['Adj Close'] / test_data[
                'Close']
            test_data.loc[:, 'High'] = test_data['High'] * test_data['Adj Close'] / train_data[
                'Close']
            test_data = test_data.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
            col_names = ['close_1_roc'] + ['close_1_roc_-{}_s'.format(i + 1) for i in range(cols_per_stock - 1)]
            if len(train_data > 0) and len(test_data > 0):
                train_wrapped_data = stockstats.wrap(train_data)[col_names]
                train_price_data = stockstats.wrap(train_data)[['close']]
                test_wrapped_data = stockstats.wrap(test_data)[col_names]
                test_price_data = stockstats.wrap(test_data)[['close']]

                train_full = train_full.join(train_wrapped_data, rsuffix=stock, how='inner')
                train_price_full = train_price_full.join(train_price_data, rsuffix=stock, how='inner')
                test_full = test_full.join(test_wrapped_data, rsuffix=stock, how='inner')
                test_price_full = test_price_full.join(test_price_data, rsuffix=stock,
                                                                   how='inner')
                count += 1
                final_stock_list.append(stock)

            if n_comp is not None and n_comp // len(stock_lists) <= count:
                break

    train_full.drop(columns=['close'], inplace=True)
    test_full.drop(columns=['close'], inplace=True)
    train_price_full.drop(columns=['close'], inplace=True)
    test_price_full.drop(columns=['close'], inplace=True)
    technicals_list = [('^VIX', 'close_1_roc'), ('^IRX', 'close'), ('^FVX', 'close')]
    technicals_data = pd.read_csv('indicators.csv', header=[0, 1], index_col=0)

    for tpl in technicals_list:
        symbol, feat = tpl

        train_data = technicals_data[download_train_start:train_end].xs(symbol, axis=1, level=1)
        test_data = technicals_data[test_start:test_end].xs(symbol, axis=1, level=1)
        
        col_names = [feat] + ['{}_-{}_s'.format(feat, i + 1) for i in range(cols_per_stock - 1)]
        train_wrapped_data = stockstats.wrap(train_data)[col_names]
        test_wrapped_data = stockstats.wrap(test_data)[col_names]
        train_full = train_full.join(train_wrapped_data, rsuffix=symbol, how='inner')
        test_full = test_full.join(test_wrapped_data, rsuffix=symbol, how='inner')

    dates = train_full.ffill().loc[train_start:].index
    train_full = train_full.ffill().loc[train_start:].to_numpy() / 100
    train_price_full = train_price_full.ffill()[train_start:].to_numpy()
    test_full = test_full.ffill().loc[test_start:].to_numpy() / 100
    test_price_full = test_price_full.ffill()[test_start:].to_numpy()

    features = len(final_stock_list) + len(technicals_list)
    train_final = np.zeros(shape=(train_full.shape[0], train_full.shape[1] // features, features))
    test_final = np.zeros(shape=(test_full.shape[0], test_full.shape[1] // features, features))

    np.savetxt(path + 'current stock_list.txt', final_stock_list, fmt='%s')

    for i in range(features):
        train_final[:, :, i] = train_full[:, i * cols_per_stock: (i+1) * cols_per_stock]
        test_final[:, :, i] = test_full[:, i * cols_per_stock: (i + 1) * cols_per_stock]

    if not return_dates:
        return train_final, train_price_full, test_final, test_price_full, final_stock_list
    else:
        return train_final, train_price_full, test_final, test_price_full, final_stock_list, dates


if __name__ == '__main__':
    assets = ['IVV', 'IJH', 'IJR', 'TLT', 'IEF', 'SHY', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA',
              'IYJ',
              'IYM', 'IYR', 'IYW', 'IDU']
    assets_no_bonds = ['IVV', 'IJH', 'IJR', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ',
                       'IYM', 'IYR', 'IYW', 'IDU']
    assets_diverse = ['IVV', 'IJH', 'IJR', 'IYK', 'IYC', 'IYH', 'IYW', 'IDU', 'GC=F', 'CL=F', 'HG=F', 'JPY=X', 'CHF=X',
                      'AUD=X', 'IEF']
    assets_div_limited = ['IVV', 'IJH', 'IJR', 'IEF', 'GC=F', 'CL=F', 'HG=F', 'JPY=X', 'CHF=X', 'AUD=X']
    assets_full_diverse = ['IEF', 'IYZ', 'IYK', 'IYC', 'IYE', 'IYF', 'IYH', 'ITA', 'IYJ', 'IYM', 'IYR', 'IYW', 'IDU',
                           'GC=F', 'CL=F', 'HG=F',
                           'ZC=F', 'KC=F']

    dow_jones = ['MMM', 'AXP', 'T', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DD', 'XOM',
                 'GE', 'GS', 'HD', 'INTC', 'IBM', 'JPM', 'JNJ', 'MCD', 'MRK', 'MSFT',
                 'NKE', 'PFE', 'PG', 'TRV', 'RTX', 'UNH', 'VZ', 'V', 'WMT', 'DIS']

    simulation = Simulation(2014, 2024, '3 pct target test/', assets=assets, rebalance=4, batch_size=64)
    simulation.simulate()

    #simulation = Simulation(2023, 2024, 'vol 256/', assets=assets, rebalance=4, batch_size=256)
    #simulation.simulate()

    #simulation = Simulation(2014, 2024, 'vol 01 corr/', assets=assets, rebalance=4, batch_size=64, t_cost=0.001)
    #simulation.simulate()

    #simulation = Simulation(2014, 2024, 'vol 0 corr/', assets=assets, rebalance=4, batch_size=64, t_cost=0)
    #simulation.simulate()

