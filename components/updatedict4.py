# import pandas_datareader as pdr
# from datetime import datetime
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import datetime as dt
# import pickle
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# import numpy as np
# import datetime as dt
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.graph_objs as go
# import plotly.plotly as py

# # dffinal = pd.read_csv('sahambaru.csv')
# dfytrain = pd.read_csv('dfytrain.csv')
# dfytest = pd.read_csv('dfytest.csv')
# dfypred = pd.read_csv('dfypred.csv')

# month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
#          'August', 'September', 'October', 'November', 'December']
# high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
# low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
# high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
# low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
# high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
# low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

# app = dash.Dash(__name__)

# server = app.server

# app.layout = html.Div([
#                 html.H1('Plot Saham', className='h1'),
#                 dcc.Graph(
#                     id='Plot1',
#                     trace0 = go.Scatter(
#                 x = month,
#                 y = high_2014,
#                 name = 'High 2014',
#                 line = dict(
#                 color = ('rgb(205, 12, 24)'),
#                 width = 4)
#                 ),
#                 trace1 = go.Scatter(
#                 x = month,
#                 y = low_2014,
#                 name = 'Low 2014',
#                 line = dict(
#                 color = ('rgb(22, 96, 167)'),
#                 width = 4,)
#                 ),
#                 trace2 = go.Scatter(
#                 x = month,
#                 y = high_2007,
#                 name = 'High 2007',
#                 line = dict(
#                 color = ('rgb(205, 12, 24)'),
#                 width = 4,
#                 dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
#                 ),
#                 trace3 = go.Scatter(
#                 x = month,
#                 y = low_2007,
#                 name = 'Low 2007',
#                 line = dict(
#                 color = ('rgb(22, 96, 167)'),
#                 width = 4,
#                 dash = 'dash')
#                 ),
#                 trace4 = go.Scatter(
#                 x = month,
#                 y = high_2000,
#                 name = 'High 2000',
#                 line = dict(
#                 color = ('rgb(205, 12, 24)'),
#                 width = 4,
#                 dash = 'dot')
#                 ),
#                 trace5 = go.Scatter(
#                 x = month,
#                 y = low_2000,
#                 name = 'Low 2000',
#                 line = dict(
#                 color = ('rgb(22, 96, 167)'),
#                 width = 4,
#                 dash = 'dot')
#                 ),
#                 data = [trace0, trace1, trace2, trace3, trace4, trace5],

#                 # Edit the layout
#                 layout = dict(title = 'Average High and Low Temperatures in New York',
#                         xaxis = dict(title = 'Month'),
#                         yaxis = dict(title = 'Temperature (degrees F)'),
#                         ),

#                 fig = dict(data=data, layout=layout),
#                 py.iplot(fig, filename='styled-line')
#                 )
#             ])

# if __name__ == '__main__':
#     # app.run_server(debug=True, port=2019)
#     app.run_server(debug=True, port=100)

