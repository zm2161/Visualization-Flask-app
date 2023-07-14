import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from pairTradingVisual2 import *
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from dash.exceptions import PreventUpdate

strategies = ["sigma", "quantile", "arima", "random forest"]



def get_options(lst):
    return ([{'label': l, 'value': l} for l in lst])


def get_levels():
    levels = {}
    levels['level_1'] = get_options(Price.columns.to_list())
    levels['level_2'] = get_options(strategies)
    return (levels)


def check_next(values):
    '''
    here I want to check if the selected level is a terminal selection
    This returns a tupple:
        True if next level has options
        True if the current level is a terminal option

    values is a list of [value/None]
    '''
    print(values)
    val1 = values[0]
    val2 = values[1]
    if val1 in Price.columns.to_list():
        if val2 in strategies:
            return True, True
        else:
            return True, False
    else:
        return False, False


DROPDOWN_OPTIONS = get_levels()

# %%
def get_home():
    # contents=html.Div([html.Center(html.H1('Pair Trading')),
    # html.Ul(
    #         [
    #             html.Li('Stock Overview'),
    #             html.Ul(
    #                 [html.Li('Choose your stock pool'),
    #                 html.Li('Stock Price time series in clusters'),
    #                 ]),
    #             html.Li('Backtesting Result'),
    #             html.Ul(
    #                 [html.Li('ARIMA'),
    #                 html.Li('Random Forest'),
    #                 ]),
    #         ]
    #     ),])
    contents = html.Div([
        html.Center(html.H1('Pair Trading')),
        # This is new, grid placement using bootstrap
        dbc.Card(

            dbc.CardBody([

                dbc.Row(
                    [
                        # dbc.Col(width = True),
                        dbc.Col(
                            dbc.Nav([
                                dbc.NavLink("1.Stock Overview", href="/return", id="return_pg")

                            ])),
                        # dbc.Col(width = True)
                    ]
                ),
                dbc.Row([
                    # dbc.Col(width = True),
                    dbc.Col(dbc.Label('1.1 Stock Time Series'), width=4),
                    dbc.Col(dbc.Label('1.2 Stock Price Volatility'), width=4),
                    # dbc.Col(width = True)
                ]),
                dbc.Row([
                    # dbc.Col(width = True),
                    dbc.Col(dbc.Nav(dbc.NavLink("2. Backtesting Result", href="/backtest",
                                                id="backtest_pg"))),
                    # dbc.Col(width = True)
                ]),
            ], id='home_inputs'), className='inputs_card'
        ),
    ])
    return (contents)


def get_return():
    contents = html.Div(
        [
            html.Center(html.H4('Stock Time Series')),
            html.Br(),
            html.Br(),
            html.Div('''
             Please input stock symbol and time series ''',
                     style={
                         'width': '60%',

                     }
                     ),

            dcc.Input(
                id="my-input",
                type="text",
                placeholder="Please input stock symbol name Default AAPL: ",
                style={"width": "40%"}
            ),

            html.Br(),

            dcc.Dropdown(id='dropdown',
                         options=[{'label': 'max', 'value': 'max'},
                                  {'label': '1y', 'value': '1y'},
                                  {'label': '6mo', 'value': '6mo'},
                                  {'label': '1mo', 'value': '1mo'},
                                  {'label': '5d', 'value': '5d'}],
                         placeholder="Please input time period Default max",
                         style={"width": "40%"}
                         ),
            html.Button(id='submit-button', n_clicks=0, children='Submit'),
            html.Div([], id="time-series-graph"),

            html.Br(),
            html.Div([], id="bar-graph"),
        ], style={'border-style': 'solid', "height": '100bp'})
    return (contents)


def get_backtest():
    contents = html.Div([
        html.Center(html.H4('Backtest Result')),
        html.Br(),
        html.Br(),
        html.Div('''
             Please input stock symbol and backtest strategy ''',
                 style={
                     'width': '60%',
                 }
                 ),

        html.Div(
            [
                dcc.Dropdown(
                    options=opt,
                    placeholder='Select level {}'.format(name.replace('_', ' ')),
                    # now I am making only the very first dropdown enabled
                    disabled=False if name == 'level_1' else True,
                    searchable=True,
                    multi=False,
                    id='{}_dd'.format(name),
                    style={'width': '10em'}
                ) for name, opt in DROPDOWN_OPTIONS.items()
            ], style={'display': 'flex', 'justify-content': 'space-between'}
        ),

        html.Br(),
        html.Div(id='alert'),
        #html.Div([], id="PnL-graph"),
        dcc.Loading(
            html.Div(id="PnL-graph"),
            id="graph1",
            # type="default",
            # type="graph",
            # type="cube",
            # type="circle",
            type="dot",
        ),
        html.Br(),
        #html.Div([], id="cluster-graph"),
        dcc.Loading(
            html.Div(id="cluster-graph"),
            id="graph1",
            # type="default",
            # type="graph",
            # type="cube",
            # type="circle",
            type="dot",
        ),

        html.Br(),
        
        html.Br(),

        html.Div([dbc.Checklist(
            options=[
                {"label": "Showing More Result", "value": 1},
            ],
            value=[],
            id="format",
            inline=True,
            switch=True,
        ),
            html.Br(),

            html.Div(id='result'),
            html.Br(),
            html.Br()
        ], style={'border-style': 'solid', "height": '100bp'}),
    ], style={'border-style': 'solid', "height": '100bp'})
    return (contents)


# %% Dash app
app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    prevent_initial_callbacks=True,
)

home_button = dbc.NavItem(dbc.NavLink('Home', href="/home", external_link=True))
return_button = dbc.NavItem(dbc.NavLink('Return', href="/return", external_link=True))
backtest_button = dbc.NavItem(dbc.NavLink('Backtest', href="/backtest", external_link=True))
navbar = dbc.Navbar(
    dbc.Container(
        [

            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(dbc.Nav([home_button, return_button, backtest_button], navbar=True), id="navbar-collapse",
                         navbar=True),
        ],
    ),
    # color="rgb(42,62,66)",
    color="blue",
    dark=True,
    style={'background-color': '#191919'},

    expand='lg'
)
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        navbar,
        html.Div(id='main_div', className='main_contents'),
    ],
)

pg_ids = ["home_pg", 'return_pg', 'backtest_pg']


@app.callback(
    # [Output(f"{pg_id}", "active") for pg_id in pg_ids] +
    Output('main_div', 'children'),
    [
        Input("url", "pathname"),
    ],
)
def main_loadings(pathname):
    # this is a linux thing 

    print(pathname)

    if pathname.find("backtest") >= 0:
        return (False, False, True, get_backtest())
    elif pathname.find("return") >= 0:
        return (False, True, False, get_return())
    else:
        # because at lunch there is no page selected
        return (True, False, False, get_home())
    
    
    


@app.callback(
    Output("time-series-graph", "children"),
    Output('bar-graph', 'children'),
    Input('submit-button', 'n_clicks'),
    State("my-input", "value"),
    State('dropdown', 'value')
)
def update_graph_1(n_clicks, ticker, dropdown_value):
    if ticker in Price.columns:
        fig_time_series = dcc.Graph(figure=plot_time_series(ticker, dropdown_value))
        fig_volatility = dcc.Graph(figure=make_barchart(calculate_volatility(ticker)))
    else:
        fig_time_series = dbc.Alert("Please select a new stock", color="warning",
                                    dismissable=True, duration=5000, className='my_alert')
        fig_volatility = ''
    return fig_time_series, fig_volatility


@app.callback(
    Output("cluster-graph", "children"),
    Output('PnL-graph', 'children'),
    [
        Output('alert', 'children'),
    ] + [Output('{}_dd'.format(name), 'disabled') for name in DROPDOWN_OPTIONS.keys()],
    [Input('{}_dd'.format(name), 'value') for name in DROPDOWN_OPTIONS.keys()]
)
def update_all(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        from dash.exceptions import PreventUpdate
        print('hey!')
        raise PreventUpdate
    else:
        dropdown = ctx.triggered[0]['prop_id'].split('.')[0]

    print(args)
    p1, p2 = check_next(args)
    if p2:
        alert = 'The selection would result in a plot of {}'.format(args)
    else:
        alert = 'Please chose further options'

    if dropdown == 'level_1_dd':
        disabled = [False] + [not p1]

    if dropdown == 'level_2_dd':
        disabled = [False] * 2

    if p2:
        ticker = args[0]
        dropdown_strategy = args[1]
        fig_cluster_, PairSymbols = plot_cluster(Price, ticker)
        fig_PnL = dcc.Graph(figure=backTestStrategy(ticker, PairSymbols, dropdown_strategy)[1])
        fig_cluster = dcc.Graph(figure=fig_cluster_)
    else:
        fig_PnL = dcc.Graph()
        fig_cluster = dcc.Graph()
    return ([fig_cluster, fig_PnL, alert] + disabled)


@app.callback(
    Output('result', 'children'),
    Input('format', 'value'),
    Input("level_1_dd", "value"),
    Input('level_2_dd', 'value')
)
def format_table(formatted,ticker, dropdown_strategy):
    print(formatted)
    if formatted:
        # PairSymbols=plot_cluster(Price, ticker)
        fig_cluster_, PairSymbols = plot_cluster(Price, ticker)
        money=backTestStrategy(ticker, PairSymbols, dropdown_strategy)[0]
        
        return f"Final money: {round(money, 2)}, Stock in Pair: {PairSymbols}"
    else:
        
        return('')


# %% Run
app.config.suppress_callback_exceptions = True
app.run_server(
    debug = True,
    port=8056
)
# %%
