import dash
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import numpy as np
from utils import *
from memoization import cached
from sklearn.manifold import TSNE 
import os

@cached 
def yf_download(ticker, period="max"):
    if isinstance(ticker, list):
        if len(ticker) == 1:
            ticker = ticker[0]
    hist = yf.download(ticker, period=period)
    hist['Date'] = hist.index
    hist.index = hist.index.strftime("%Y-%m-%d")
    return hist 


@cached # Download data from yahoo finance
def download_data(ticker, period="max"):
    print("Downloading", ticker)
    hist = yf_download(ticker, period=period)
    # if len(hist) < 20: return download_data("AAPL", period)
    # hist['Date'] = hist.index
    # hist.index = hist.index.strftime("%Y-%m-%d")
    return hist
# def download_data(ticker, period="max"):
#     myTicker = yf.Ticker(ticker)
#     hist = myTicker.history(period=period).reset_index()
#     hist['Date'] = pd.to_datetime(hist['Date'], errors = 'coerce')
#     return hist


@cached # Plot time series of the ticker
def plot_time_series(ticker, period="max"):
    ticker = ticker if ticker else "AAPL"
    if isinstance(ticker, list):
        tickers = ticker 
    else:
        tickers = [ticker]
    fig = go.Figure()
    print("ticker", ticker)
    print(tickers)
    for ticker in tickers:
        print("download ticker", ticker)
        hist = download_data(ticker, period)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Open"]))
    # fig = px.scatter(hist, y="Open", x='Date')

    # The update_layout method allows us to give some formatting to the graph
    fig.update_layout(
        title_text = "Time Series Plot of {}".format(tickers) if period == "max" \
            else "Time Series Plot in a period of {} of {}".format(period, tickers),
        title_x = 0.5,
        yaxis = {
            'title': 'PnL'}
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        # stepmode="backward"
                        ),
                    dict(count=6,
                        label="6m",
                        step="month",
                        # stepmode="backward"
                        ),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        # stepmode="todate"
                        ),
                    dict(count=1,
                        label="1y",
                        step="year",
                        # stepmode="backward"
                        ),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            # type="date",
            # title='This is a date'
        )
    )
    return fig 


@cached
def calculate_volatility(ticker):
    if isinstance(ticker, list):
        tikcer = ticker[0]
    hist = download_data(ticker)
    hist["returns"] = hist["Close"].pct_change()
    hist.dropna(inplace=True)

    hist["labels"] = pd.qcut(hist["Volume"], 20)
    returns_mean = hist.groupby("labels")["returns"].mean()

    data = []
    data.append(
        go.Bar(
        name = "returns distribution",
        x = [f"{round(x.left, -len(str(x.left))+4)}-{round(x.right, -len(str(x.right))+4)}" for x in returns_mean.index.categories],
        y = returns_mean.tolist()
        )
    )
    return data


@cached
def make_barchart(data):
    fig = go.Figure(data = data)

    fig.update_layout(
        barmode = 'group',
        title = 'Bar Chart of Equity Returns grouped by Volume',
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        xaxis = dict(
            showline = True, 
            linewidth = 2, 
            linecolor = 'black'
        ),
        yaxis=dict(
            title = 'Stock Returns',
            titlefont_size = 16,
            tickfont_size = 14,
            gridcolor = '#dfe5ed'
        )
    )

    fig.layout.hovermode = 'x'
    return(fig)

@cached
def calculate_simulation(ticker,days=30, trials=100):
    if isinstance(ticker, list):
        ticker = ticker[0]
    hist = download_data(ticker)
    from scipy.stats import norm
    data=hist["Close"].pct_change()
    log_returns = np.log(1 + data)
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5*var)
    stdev = log_returns.std()

    #Z = norm.ppf(np.random.rand(days, trials)) #days, trials
    Z = np.random.randn(days, trials)
    daily_returns = np.exp(drift + stdev * Z)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = hist["Close"].iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]

    price_paths=price_paths.T
    data2 = []
    last=[]
    for path in price_paths:
        last.append(path[-1])
        data2.append(
            go.Scatter
            (
            y = [i for i in path],
            x = [i for i in range(days)]
            )
        )
    df_last=pd.DataFrame({'price':last})
    return data2, df_last

@cached
def plot_simulation(input_value):
    data, df_last = calculate_simulation(input_value)
    fig = go.Figure(data = data)
    import plotly.express as px
    histog = px.histogram(df_last, x='price')
    fig.update_layout(
        title_text = "Monte Carlo simulation of {} in 100 trials".format(input_value),
        title_x = 0.5,
        yaxis = {
            'title': 'Price'}
    )
    histog.update_layout(
        title_text = "Distribution of prices in 30days of Monte Carlo simulation of {}".format(input_value),
        title_x = 0.5,
        yaxis = {
            'title': 'Count'}
    )
    return fig, histog


@cached 
def calculate_hedged_difference(symbol1, symbol2, period="5y"):
    df = download_data([symbol1, symbol2], period)["Adj Close"].dropna()
    df["Hedged"] = df.iloc[:,0] * get_hedgeRatio(df)
    df["dif"] = df["Hedged"] - df.iloc[:,1]
    return df

@cached
def backTestStrategy(symbol1, symbol2, strategy="sigma", period="5y"):
    strategies = {"sigma": StrategySigma(), "quantile": StrategyQuantile(), "arima": StrategyArima(), "random forest": StrategyRF()}
    strategy_type = strategies[strategy]
    period, lookback = "5y", 252
    if strategy == "arima":
        period = "6mo"
        lookback = 21
    elif strategy == "random forest":
        period = "2y"
    df = calculate_hedged_difference(symbol1, symbol2, period=period)
    pnl, result = backTester(df["dif"], strategy_type, lookback=lookback)
    fig = px.line(result, x=result.index, y="PnL")
    fig.update_layout(
    title_text = "Pair Trading PnL plot of {} and {} of period {}".format(symbol1, symbol2, period),
    title_x = 0.5,
    yaxis = {
        'title': 'Price'}
    )
    return pnl, fig


@cached
def calculate_cluster(Price, TRAIN_END):
    df_ret_train = Price.pct_change()[1:].loc[:TRAIN_END].copy()
    labels = OPTICS_fit(normalize(df_ret_train))
    n_clusters_ = len(set(labels)) - 1
    print("Clusters discovered:", n_clusters_)
    clustered_series_all = pd.Series(index=df_ret_train.columns, data=labels)
    # If it is -1, it means it is a seperate cluster with very few elements
    clustered_series = clustered_series_all[clustered_series_all != -1]
    counts = clustered_series.value_counts()
    ticker_count_reduced = counts[(counts > 1)]
    print("Pairs in group to evaluate:", (ticker_count_reduced * (ticker_count_reduced - 1)).sum() // 2)
    X_tsne = TSNE(learning_rate=1000, perplexity=25).fit_transform(normalize(df_ret_train).T)
    return labels, X_tsne
    
    
@cached
def plot_cluster(Price, symbol):
    if symbol not in Price.columns:
        hist = download_data(symbol)
        hist[symbol] = hist["Adj Close"]
        Price = Price.merge(hist[symbol], left_index=True, right_index=True)
   
    labels, X_tsne = calculate_cluster(Price, TRAIN_END)
    names = Price.columns
    symbol_position = names.tolist().index(symbol)
    label = labels[symbol_position]
    # print("label", label)
    if label != -1:
        label_index = [i for i,x in enumerate(labels) if x==label]
        PairSymbols = Price.columns[label_index].tolist()
        PairSymbols.remove(symbol)
        PairSymbols = PairSymbols[0]
    else:
        PairSymbols = "AMZN"
    print("PairSymbols", PairSymbols)
    

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_tsne[(labels!=-1), 0], y=X_tsne[(labels!=-1), 1], mode="markers", marker_color=labels[labels != -1], marker_size=15, name="clustered", text=names[labels!=-1]))
    fig.add_trace(go.Scatter(x=X_tsne[(labels==-1), 0], y=X_tsne[(labels==-1), 1], mode="markers", marker_size=5, name="unclustered", text=names[labels==-1]))
    fig.add_trace(go.Scatter(x=X_tsne[labels==label, [0]], y=X_tsne[labels==label, [1]], mode="markers", marker_size=20, marker_color= label if label != -1 else None, name=f"clustered with {symbol}", text=names[labels==label])) if label != -1 \
        else fig.add_trace(go.Scatter(x=X_tsne[symbol_position, [0]], y=X_tsne[symbol_position, [1]], mode="markers", marker_size=20, marker_color= label if label != -1 else None, name=symbol))
    fig.update_layout(
        title_text = "Cluster result of OPTICS Cluster",
        title_x = 0.5,
    )

    return fig, PairSymbols

#os.chdir('/Users/mazhuoran/Desktop/6191/Visualization6191')
if os.path.exists("website/stock_5years.csv"):
    Price = pd.read_csv("website/stock_5years.csv", index_col=0)
else:
    Price = pd.read_csv("stock_5years.csv", index_col=0)

TRAIN_START = "2017-01-01"
TRAIN_END = "2020-01-01"
ticker = "KSS"
strategies = ["sigma", "quantile", "arima", "random forest"]
# labels, X_tsne = calculate_cluster(Price, TRAIN_END)

if __name__ == "__main__":    
    hist = download_data(ticker)
    fig_time_series = plot_time_series(ticker)
    fig_time_series_period = plot_time_series(ticker, "1y")
    fig_volatility = make_barchart(calculate_volatility(ticker))
    fig_simulation, fig_simulation_histogram = plot_simulation(ticker)

    # fig_cluster, PairSymbols = plot_cluster(Price, ticker)
    fig_cluster, PairSymbols = plot_cluster(Price, ticker)
    _, fig_PnL = backTestStrategy(ticker, PairSymbols)
    
else:
    fig_time_series = fig_time_series_period = fig_volatility = fig_simulation = fig_simulation_histogram = fig_cluster = fig_PnL = go.Figure()

#%% Dash app
app = Dash(
    prevent_initial_callbacks = True
)

app.layout = html.Div(
    [
        
        html.Center(html.H4('My Second Dash App - Yey!!!')),
        html.Br(),
        html.Br(),

        dcc.Input(
            id="my-input",
            type="text",
            placeholder="Please input stock symbol name Default AAPL: ",
            style={ "width": "20%"}
        ),
        dcc.Graph(id='sim_plot', figure=fig_simulation),
        dcc.Graph(id='sim_hist', figure=fig_simulation_histogram),
        
        html.Br(),
        html.Br(),
        html.Label('Input time period: '),
        dcc.Dropdown(id='dropdown',
        options=[{'label': 'max', 'value': 'max'},
                {'label': '1y', 'value': '1y'},
                {'label': '6mo', 'value': '6mo'},
                {'label': '1mo', 'value': '1mo'},
                {'label': '5d', 'value': '5d'}],
            placeholder = 'max'
            #options=[{'labels': comp['label'], 'value':comp['label']} for comp in comp_options]
        ),
        #html.H4('Price graph of different period', style = {'text-align': 'center', 'color':'blue','font-weight': 'bold'}),
        # multiple line of text
        
        
        dcc.Graph(id='mul_plot', figure=fig_time_series_period),

        html.Br(),
        html.Div(id='my-output'),


        html.Div('''
             This app displays a graph of the entire price history of {}.'''.format(ticker),
             style = {
                 'width': '60%',
                 'text-align': 'center',
                 'margin-left': 'auto',
                 'margin-right': 'auto',
             }
        ),
        
        dcc.Graph(figure = fig_time_series, id="graphic"),

        html.Br(),
        dcc.Graph(figure = fig_volatility, id="bar"),
        html.Br(),
        
        html.Label('Input Strategy: default sigma'), 
        
        dcc.Dropdown(
            id='dropdown_strategy',
            options=[{"label": x, "value": x} for x in strategies],
            placeholder = "sigma"
        ),
 
        html.Br(),
        dcc.Graph(id="PnL_plot", figure=fig_PnL),
        html.Br(),
        dcc.Graph(id="fig_cluster", figure=fig_cluster),
        
    ],  #I could also put the list comprehension here
    style ={
        'margin': '2em',
        'border-radius': '1em',
        'border-style': 'solid', 
        'padding': '2em',
        'background': '#ededed'
    }
)


@app.callback(
              Output(component_id="graphic", component_property="figure"),
              Output(component_id="bar", component_property="figure"),
              Output(component_id='mul_plot', component_property='figure'),
              Output(component_id='sim_plot', component_property='figure'),
              Output(component_id='sim_hist', component_property='figure'),
              Output(component_id='PnL_plot', component_property='figure'),
              Output(component_id="fig_cluster", component_property="figure"),
              Input(component_id="my-input", component_property="value"),
              Input(component_id='dropdown', component_property='value'),
              Input(component_id='dropdown_strategy', component_property='value'),
              )
def update_all(ticker, dropdown_value, dropdown_strategy):
    ticker = ticker if ticker else "AAPL"
    dropdown_value = dropdown_value if dropdown_value else "max"
    dropdown_strategy = dropdown_strategy if dropdown_strategy else "sigma"
    tickers = ticker.replace(" ", "").split(",")
    ticker = tickers[0]
    print("update_all", ticker, "tickers", tickers)
    fig_time_series = plot_time_series(tickers)
    fig_volatility = make_barchart(calculate_volatility(ticker))
    fig_mul_plot = plot_time_series(ticker,dropdown_value)
    figure_simulation = plot_simulation(ticker)
    fig_cluster, PairSymbols = plot_cluster(Price, ticker)
    symbol1 = tickers[0] if len(tickers) == 2 and tickers[0] else ticker
    symbol2 = tickers[1] if len(tickers) == 2 and tickers[1] else PairSymbols
    fig_PnL = backTestStrategy(symbol1, symbol2, dropdown_strategy)[1]
    return (fig_time_series, fig_volatility, fig_mul_plot, *figure_simulation, fig_PnL, fig_cluster)


if __name__ == "__main__":
    print('About to start...')
                    
    app.run_server(
        debug = True,
        port = 8062
    )

# simulation(ticker="AAPL",days=30, trials=100)
