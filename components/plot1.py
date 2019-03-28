import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd

color_set = ['#80aaff','#cc0000']

def legend(val) :
    if(val == 'red') :
        return 'train'
    elif(val == 'blue') :
        return 'Machine learning'
    elif(val == 'green'):
        return 'Test'

saham = pd.read_csv('saham3.csv')
# df1=pd.read
symbol = (saham['symbol'][1])
# print(symbol)

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
                html.H1('Plot SAHAM with Machine Learning  '+str(symbol) , className='h1'),
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
