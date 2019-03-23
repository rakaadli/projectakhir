import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

color_set = ['#80aaff','#cc0000']

# def legend(val) :
#     if(val == 2) :
#         return 'indica'
#     elif(val == 1) :
#         return 'sativa'
#     else :
#         return 'hybrid'

def renderlineplot(df) :
    return html.Div([
                html.H1('Plot Saham', className='h1'),
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
                html.H1('Plot SAHAM SAHAM', className='h1'),
                dcc.Graph(
                    id='Plot2',
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
