
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd


   

saham = pd.read_csv('saham3.csv')
# df1=pd.read
symbol = (saham['symbol'][1])
# print(symbol)
saham1 = pd.read_csv('saham5.csv')
# df1=pd.read
symbol1 = (saham['symbol'][1])
saham2plot = pd.read_csv('saham2plot.csv')
saham2plotML = pd.read_csv('saham2plotML.csv')

color_set = ['#80aaff','#cc0000']

def legend(val) :
    if(val == 'red') :
        return 'train'
    elif(val == 'blue') :
        return 'Machine learning'
    elif(val == 'green'):
        return 'Test'

def legend1(val) :
    if(val == str(saham2plot['symbol'][1])) :
        return str(saham2plot['symbol'][1])
    elif(val == str(saham2plot['symbol'][int(len(saham2plot['symbol'])-1)])) :
        return str(saham2plot['symbol'][int(len(saham2plot['symbol'])-1)])

def legend2(val) :
    if(val == str(saham2plotML['symbol'][1])) :
        return str(saham2plotML['symbol'][1])
    elif(val == str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])) :
        return str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])

def legend3(val,val1) :
    if(val == 'red' and val1 == str(saham2plotML['symbol'][1])) :
        return 'train ' + str(saham2plotML['symbol'][1])
    elif(val == 'blue' and val1 == str(saham2plotML['symbol'][1])) :
        return 'Machine learning ' + str(saham2plotML['symbol'][1])
    elif(val == 'green' and val1 == str(saham2plotML['symbol'][1])):
        return 'Test ' + str(saham2plotML['symbol'][1])
    elif(val == 'red' and val1 == str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])):
        return 'Train ' + str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])
    elif(val == 'blue' and val1 == str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])):
        return 'Machine Learning ' + str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])
    elif(val == 'green' and val1 == str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])):
        return 'Test ' + str(saham2plotML['symbol'][int(len(saham2plotML['symbol'])-1)])

def renderlineplot(df) :
    return html.Div([
                html.H1('Plot Saham '+str(df['symbol'][1]), className='h1'),
                dcc.Graph(
                    id='Plot1',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df['Date'],
                                y=df['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                # name=legend(col)
                            ) 
                            # for col in df['Type'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])


def renderlineplot1(df) :
    return html.Div([
                html.H1('Plot SAHAM with Machine Learning  '+str(df['ML'][1])+' '+str(symbol1) , className='h1'),
                dcc.Graph(
                    id='Plot2',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df[df['warna'] == col]['Date'],
                                y=df[df['warna'] == col]['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                name=legend(col)
                            ) 
                            for col in df['warna'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])

def renderlineplot2(df) :
    return html.Div([
                html.H1('Plot Saham '+str(df['symbol'][1]), className='h1'),
                dcc.Graph(
                    id='Plot3',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df['Date'],
                                y=df['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                # name=legend(col)
                            ) 
                            # for col in df['Type'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])


def renderlineplot3(df) :
    return html.Div([
                html.H1('Plot SAHAM with Machine Learning  '+str(df['ML'][1])+' '+str(symbol1) , className='h1'),
                dcc.Graph(
                    id='Plot4',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df[df['warna'] == col]['Date'],
                                y=df[df['warna'] == col]['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                name=legend(col)
                            ) 
                            for col in df['warna'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])



def renderline2plot(df) :
    return html.Div([
                html.H1('Plot Saham '+str(df['symbol'][1])+ ' dan '+ str(df['symbol'][int(len(df['symbol'])-1)]), className='h1'),
                dcc.Graph(
                    id='Plot5',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df[df['symbol'] == col]['Date'],
                                y=df[df['symbol'] == col]['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                name=legend1(col)
                            )  for col in df['symbol'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])

# ['symbol']
def renderline2plotML(df) :
    return html.Div([
                html.H1('Plot SAHAM with Machine Learning  '+str(df['ML'][1])+' '+str(symbol1) + ' dan ' +str(df['ML'][int(len(df['symbol'])-1)]+' '+str(df['symbol'][int(len(df['symbol'])-1)])) , className='h1'),
                dcc.Graph(
                    id='Plot6',
                    figure={
                        'data': [
                            go.Scatter(
                                x=df[df['warna'] == col][df['symbol'] == col1]['Date'],
                                y=df[df['warna']== col][df['symbol'] == col1]['Close'],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                name=legend3(col,col1)
                            ) 
                            for col in df['warna'].unique()
                            for col1 in df['symbol'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])


def renderplotbacktest1(dfbacktest1):
    return html.Div([
                html.H1('Plot profit/loss hasil backtest' , className='h1'),
                dcc.Graph(
                    id='plot7',
                    figure={
                        'data': [
                            go.Scatter(
                                # x=df[df['warna'] == col][df['symbol'] == col1]['Date'],
                                # y=df[df['warna']== col][df['symbol'] == col1]['Close'],
                                x=dfbacktest1['Unnamed: 0'],
                                y=dfbacktest1["End Port. Value"],
                                mode='lines',
                                # marker=dict(color=color_set[i], size=10, line=dict(width=0.5, color='white')),
                                # name=legend3(col,col1)
                            ) 
                            # for col in df['warna'].unique()
                            # for col1 in df['symbol'].unique()
                        ],
                        'layout': go.Layout(
                            xaxis= dict(title='Actual Harga saham'),
                            yaxis={'title': 'Price'},
                            margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
                            hovermode='closest'
                        )
                    }
                )
            ])
