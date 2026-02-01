import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import matplotlib.cm as cm
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import warnings
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')
#new

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN , OPTICS
from sklearn.manifold import TSNE 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

#new change
def PCA_normalize(df_returns, n_components=50):
    df_ret_train_normalized = preprocessing.StandardScaler().fit_transform(df_returns)
    pca = PCA(n_components=n_components)
    pca.fit(df_ret_train_normalized)
    X = pca.components_.T    
    return X 

def normalize(df_returns):
    # normalize the different stock return of one day
    df_ret_train_normalized = preprocessing.StandardScaler().fit_transform(df_returns.T)
    return df_ret_train_normalized.T

def DBSCAN_fit(df_returns, eps=2, min_samples=2):
    X = df_returns.T
    clf = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clf.labels_
    return labels

def OPTICS_fit(df_returns, min_samples=2):
    X = df_returns.T
    clf = OPTICS(min_samples=min_samples, xi=0.01).fit(X)
    labels = clf.labels_
    return labels


def get_hedgeRatio(df):
    """
    get PCA hedge Ratio
    """
    pca = PCA().fit(df)
    hedgeRatio = pca.components_[0][1] / pca.components_[0][0]
    return hedgeRatio


def Cointegration(Price, cluster, significance, start_day, end_day): 
    pair_coin = []
    p_value = []
    n = cluster.shape[0] 
    keys = cluster.keys() 
    for i in range(n):
        for j in range(i+1,n):
            asset_1 = Price.loc[start_day:end_day, keys[i]] 
            asset_2 = Price.loc[start_day:end_day, keys[j]] 
            results = sm.OLS(asset_1, asset_2).fit() 
            predict = results.predict(asset_2)
            error = asset_1 - predict
            ADFtest = ts.adfuller(error)
            if ADFtest[1] < significance:
                pair_coin.append([keys[i], keys[j]])
                p_value.append(ADFtest[1]) 
    return p_value, pair_coin


def PairSelection(Price, ticker_count_reduced, clustered_series, significance, start_day, end_day, E_selection):
    Opt_pairs = [] # to get best pair in cluster i
    if E_selection == True: # select one pair from each cluster 
        for i in range(len(ticker_count_reduced)):
            cluster = clustered_series[clustered_series == i]
            keys = cluster.keys()
            result = Cointegration(Price, cluster, significance, start_day, end_day) 
            if len(result[0]) > 0:
                if np.min(result[0]) < significance:
                    index = np.where(result[0] == np.min(result[0]))[0][0] 
                    Opt_pairs.append([result[1][index][0], result[1][index][1]])
    else:
        p_value_contval = []
        pairs_contval = []
        for i in range(len(ticker_count_reduced)):
            cluster = clustered_series[clustered_series == i]
            keys = cluster.keys()
            result = Cointegration(Price, cluster, significance, start_day, end_day)

            if len(result[0]) > 0: 
                p_value_contval += result[0] 
                pairs_contval += result[1]
        Opt_pairs = [x for _, x in sorted(zip(p_value_contval, pairs_contval))]
    
    return Opt_pairs


class BaseStrategy:
    def __init__(self):
        self.prices_hist = []
        self.long_threshold = 0
        self.long_stop_threshold = 0
        self.short_threshold = 0
        self.short_stop_threshold = 0
        self.model = None 
    
    def fit(self, prices_hist : pd.Series):
        pass 

    def predict(self):
        return (self.long_threshold, 
                self.long_stop_threshold, 
                self.short_threshold, 
                self.short_stop_threshold,)


class StrategyArima(BaseStrategy):
    def __init__(self, params=[-0.5, 0, 0.5, 0]):
        super().__init__()
        self.params = params
        self.model = None
    
    def fit(self, prices_hist, pdq=(5,1,0)):
        assert len(pdq) == 3, "wrong length of pdq"
        p, d, q = pdq
        model = ARIMA(prices_hist, order=(p,d,q)).fit()
        output = model.forecast()
        yhat = output.iloc[0]
        print("Arima predicted", output.iloc[0])
        sigma = prices_hist.std()
        l, ls, s, ss = self.params
        self.long_threshold = yhat + l * sigma 
        self.long_stop_threshold = yhat + ls * sigma 
        self.short_threshold = yhat + s * sigma 
        self.short_stop_threshold = yhat + ss * sigma 


class StrategyQuantile(BaseStrategy):
    def __init__(self, params=[0.2, 0.5, 0.8, 0.5]):
        super().__init__()
        self.params = params
    
    def fit(self, prices_hist):
        l, ls, s, ss = self.params
        self.long_threshold = prices_hist.quantile(l)
        self.long_stop_threshold = prices_hist.quantile(ls)
        self.short_threshold = prices_hist.quantile(s)
        self.short_stop_threshold = prices_hist.quantile(ss)


class StrategySigma(BaseStrategy):
    def __init__(self, params=[-1, 0, 1, 0]):
        super().__init__()
        self.params = params
    
    def fit(self, prices_hist):
        sigma = prices_hist.std()
        mu = prices_hist.mean()
        l, ls, s, ss = self.params
        self.long_threshold = mu + l * sigma
        self.long_stop_threshold = mu + ls * sigma
        self.short_threshold = mu + s * sigma
        self.short_stop_threshold = mu + ss * sigma


class StrategyRF(BaseStrategy):
    def __init__(self, params=[-0.5, 0, 0.5, 0]):
        super().__init__()
        self.params = params
        self.clf = None

    def fit(self, prices_hist):
        if not self.clf or len(prices_hist) % 20 == 0:
            difs = pd.DataFrame(prices_hist)
            difs.columns = ["dif"]
            for i in range(1,10):
                difs[f"dif{i}"] = difs["dif"].shift(i)
            difs = difs.dropna()
            clf = RandomForestRegressor()
            X_train, y_train = difs.iloc[:-1, 1:], difs.iloc[:-1, 0]
            X_test, y_test = difs.iloc[-1, 1:], difs.iloc[-1:, 0]
            clf.fit(X_train, y_train)
            self.clf = clf 
        X_test = prices_hist[-9:][::-1]
        yhat = self.clf.predict([X_test])[0]
        print("Random Forest predicted", yhat)
        sigma = prices_hist.std()
        l, ls, s, ss = self.params
        self.long_threshold = yhat + l * sigma 
        self.long_stop_threshold = yhat + ls * sigma 
        self.short_threshold = yhat + s * sigma 
        self.short_stop_threshold = yhat + ss * sigma 
        


def backTester(prices : pd.Series, strategy: BaseStrategy, lookback=252*3):
    
    prices_train, prices_test = prices[:lookback], prices[lookback:]

    money = 0; long = False; short = False
    hists = [] # store each transaction information
    history = prices_train.tolist()
    for index, price in prices_test.iteritems():
        strategy.fit(pd.Series(history))
        long_threshold, long_stop_threshold, \
            short_threshold, short_stop_threshold = strategy.predict()
        history.append(price)
        if price < long_threshold and not long:
            long = True
            money -= price 
        if price > long_stop_threshold and long:
            long = False 
            money += price 
        if price > short_threshold and not short:  
            short = True 
            money += price 
        if price < short_stop_threshold and short:
            short = False 
            money -= price 
        hists.append([index, price, long, short, money])

    # sell or buy back on the last day
    if short: money -= price 
    if long: money += price 

    print("Final money", round(money, 2))
    df_hist = pd.DataFrame(hists, columns=["date", "price_dif", "long", "short", "money"])
    df_hist["PnL"] = df_hist["money"] + (df_hist["long"].astype(int) - df_hist["short"].astype(int)) * df_hist["price_dif"]
    
    return money, df_hist
