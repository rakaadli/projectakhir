import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pickle
from components.Table1 import renderTable
from components.plot1 import renderlineplot, renderlineplot1
from components.modelPredict2 import renderModelPredict
from components.updateDict2 import updateDict2, updateDict3


app = dash.Dash(__name__)

server = app.server

saham = pd.read_csv('saham3.csv')
saham1 = pd.read_csv('sahambaru.csv')
loadModel = pickle.load(open('logreg.sav', 'rb'))
app.title = 'Dashboard Saham'


app.layout = html.Div(children=[
    html.H1(children='Dashboard Saham (by Raka Adli))',className='titleDashboard'),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Table Stocks Dataset', value='tab-1',children=[
            renderTable(saham)
        ]),
        dcc.Tab(label='Stocks Line Plot', value='tab-2',children=[
            renderlineplot(saham),
            renderlineplot1(saham1)
        ]),
        dcc.Tab(label='Test Predict', value='tab-3',children=[
            renderModelPredict()
        ]),
    ], style={
        'fontFamily': 'system-ui'
    }, content_style={
        'fontFamily': 'Arial',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
    })
], style={
    'maxWidth': '1200px',
    'margin': '0 auto'
})

@app.callback(
    Output('table-multicol-sorting', "data"),
    [Input('table-multicol-sorting', "pagination_settings"),
     Input('table-multicol-sorting', "sorting_settings")])

def update_graph(pagination_settings, sorting_settings):
    # print(sorting_settings)
    if len(sorting_settings):
        dff = saham.sort_values(
            [col['column_id'] for col in sorting_settings],
            ascending=[
                col['direction'] == 'asc'
                for col in sorting_settings
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = saham

    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('rows')

@app.callback(
    Output('outputPredict', 'children'),
    [Input('buttonPredict', 'n_clicks')],
    [State('datein', 'value'), 
    State('dateout', 'value'),
    State('monthin', 'value'),
    State('monthout', 'value'),
    State('yearin', 'value'),
    State('yearout', 'value'),
    State('train', 'value'),
    State('test', 'value'),
    State('symbol', 'value')
      ])


def update_output(n_clicks,datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol) :
    
    datastocks1 = updateDict2(datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol)

    datastockslogreg= updateDict3(train,test)

    # data = [datastocks1['Date'],datastocks1['Close']]
    

    # prediction = loadModel.predict(data)
    # predictProba = loadModel.predict_proba(data)
    # # hasil = ''
    # print(prediction)
    # print(predictProba)

    # data = datastocks1['Datecadangan', 'Close']
    # print(data)
    # for i in datastocks1.values():
    #     data.append(i)
    # print(len(data))
    # datastocks1.to_csv('saham3.csv')
    # hasil= loadModel.accuracy()
    # predik = loadModel.accuracy()
    hasil= ''
    predik = ''
    # saham = pd.read_csv('saham.csv')
    # print(data)
    # return print(datastocks1)
    return ('Prediction : ' + hasil + " | Predict Proba : " + str(predik) ) + "silahkan refresh ulang atau tunggu sekitar 30 detik"

from os import path

extra_dirs = ['./',]
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in os.walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

if __name__ == '__main__':
    # app.run_server(debug=True, port=2019)
    app.run_server(debug=True, port=2019, extra_files = extra_files)