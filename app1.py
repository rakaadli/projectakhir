
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pickle
from components.Table1 import renderTable,renderTablesignal1, renderTablebacktest1
from components.plot1 import renderlineplot, renderlineplot1, renderlineplot2, renderlineplot3, renderline2plot, renderline2plotML, renderplotbacktest1
from components.modelPredict2 import renderModelPredict,renderModelPredict1,renderModelPredict2
from components.updateDict2 import updateDict2, updateDict3, updateDict21, updateDict31,update2plotsaham, update2plotsahamML, datastocksbacktestsaham1
import dash_table



app = dash.Dash(__name__)

server = app.server

saham = pd.read_csv('saham3.csv')
saham1 = pd.read_csv('sahambaru.csv')
saham2 = pd.read_csv('saham5.csv')
saham3 = pd.read_csv('sahambaru1.csv')
saham2plot = pd.read_csv('saham2plot.csv')
saham2plotML = pd.read_csv('saham2plotML.csv')
dfbacktest1 = pd.read_csv('datastocks_backtest.csv')

# loadModel = pickle.load(open('linreg.sav', 'rb'))
app.title = 'Dashboard Saham'


app.layout = html.Div(children=[
    html.H1(children='Dashboard Saham (by Raka Adli))',className='titleDashboard'),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Table Stocks Dataset', value='tab-1',children=[
            renderTable(saham)
            # renderTable1(saham2),
            # renderTable2(saham)
            # renderTable(saham3)


        ]),
        dcc.Tab(label='Stocks Line Plot', value='tab-2',children=[
            renderlineplot(saham),
            renderlineplot1(saham1),
            renderlineplot2(saham2),
            renderlineplot3(saham3),
            renderline2plot(saham2plot),
            renderline2plotML(saham2plotML)
        ]),
        dcc.Tab(label='Test Predict', value='tab-3',children=[
            renderModelPredict(),
            renderModelPredict1(),
            renderModelPredict2()
        ]),
        dcc.Tab(label='Trading signal and Backtest + Profit/loss', value='tab-4',children=[
            renderTablesignal1(),
            renderTablebacktest1(),
            renderplotbacktest1(dfbacktest1)
            
        ])
        
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

# @app.callback(
#     Output('table-multicol-sorting', "data"),Output('table-multicol-sorting1', 'data')
#     [Input('table-multicol-sorting', "pagination_settings"),
#      Input('table-multicol-sorting', "sorting_settings"),
#      Input('table-multicol-sorting1', "pagination_settings"),
#      Input('table-multicol-sorting1', "sorting_settings")])

@app.callback(
    Output('table-multicol-sorting', "data"),
    [Input('table-multicol-sorting', "pagination_settings"),
     Input('table-multicol-sorting', "sorting_settings")])

# @app.callback(
#     Output('table-multicol-sorting1', "data"),
#     [Input('table-multicol-sorting1', "pagination_settings"),
#      Input('table-multicol-sorting1', "sorting_settings")])

# @app.callback(
#     Output('table-multicol-sorting2', "data"),
#     [Input('table-multicol-sorting2', "pagination_settings"),
#      Input('table-multicol-sorting2', "sorting_settings")])

# @app.callback(
#     Output('table-multicol-sorting1', "data"),
#     [Input('table-multicol-sorting1', "pagination_settings"),
#      Input('table-multicol-sorting1', "sorting_settings")])


def update_graph1(pagination_settings, sorting_settings):
    # print(sorting_settings)
    if len(sorting_settings):
        dff = saham2.sort_values(
            [col['column_id'] for col in sorting_settings],
            ascending=[
                col['direction'] == 'asc'
                for col in sorting_settings
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = saham2

    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('rows')

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

# active_cell, column_conditional_dropdowns, column_conditional_tooltips, column_static_dropdown, column_static_tooltip, columns, content_style, css, data, data_previous, data_timestamp, derived_viewport_data, derived_viewport_indices, derived_viewport_selected_rows, derived_virtual_data, derived_virtual_indices, derived_virtual_selected_rows, dropdown_properties, editable, end_cell, filtering, filtering_settings, filtering_type, filtering_types, id, is_focused, locale_format, merge_duplicate_headers, n_fixed_columns, n_fixed_rows, navigation, pagination_mode, pagination_settings, row_deletable, row_selectable, selected_cells, selected_rows, sorting, sorting_settings, sorting_treat_empty_string_as_none, sorting_type, start_cell, style_as_list_view, style_cell, style_cell_conditional, style_data, style_data_conditional, style_filter, style_filter_conditional, style_header, style_header_conditional, style_table, tooltip_delay, tooltip_duration, tooltips, virtualization

# def update_graph1(pagination_settings1, sorting_settings1):
#     # print(sorting_settings)
#     if len(sorting_settings1):
#         dff = saham.sort_values(
#             [col['column_id'] for col in sorting_settings1],
#             ascending=[
#                 col['direction'] == 'asc'
#                 for col in sorting_settings1
#             ],
#             inplace=False
#         )
#     else:
#         # No sort is applied
#         dff = saham

#     return dff.iloc[
#         pagination_settings1['current_page']*pagination_settings1['page_size']:
#         (pagination_settings1['current_page'] + 1)*pagination_settings1['page_size']
#     ].to_dict('rows')


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
    State('symbol', 'value'),
    State('ML', 'value')
      ])


def update_output(n_clicks,datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol,ML) :
    
    
    datastocks1 = updateDict2(datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol)
    datastocksML= updateDict3(train,test,ML,symbol)
    dffinal = update2plotsaham(symbol)
    dffinal1 = update2plotsahamML(ML,symbol)
    # datastocksbacktestsaham = datastocksbacktestsaham1(datein, dateout ,monthin,monthout,yearin,yearout,symbol)
    



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
    return ('Prediction : ' + rms + " | Predict Proba : " + str(predik) ) + "silahkan refresh ulang atau tunggu sekitar 30 detik"

@app.callback(
    Output('outputPredict1', 'children'),
    [Input('buttonPredict1', 'n_clicks')],
    [State('datein1', 'value'), 
    State('dateout1', 'value'),
    State('monthin1', 'value'),
    State('monthout1', 'value'),
    State('yearin1', 'value'),
    State('yearout1', 'value'),
    State('train1', 'value'),
    State('test1', 'value'),
    State('symbol1', 'value'),
    State('ML1', 'value')
      ])

def update_output1(n_clicks,datein1, dateout1 ,monthin1,monthout1,yearin1,yearout1,train1,test1,symbol1,ML1) :
    
    datastocks11 = updateDict21(datein1, dateout1,monthin1,monthout1,yearin1,yearout1,train1,test1,symbol1)

    rms= updateDict31(train1,test1,ML1,symbol1)

    # data = [datastocks1['Date'],datastocks1['Close']]

    dffinal = update2plotsaham(symbol1)
    dffinal1 = update2plotsahamML(ML1,symbol1)

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
    return ('Prediction : ' + rms + " | Predict Proba : " + str(predik) ) + "silahkan refresh ulang atau tunggu sekitar 30 detik"

@app.callback(
    Output('outputPredict2', 'children'),
    [Input('buttonPredict2', 'n_clicks')],
    [State('datein2', 'value'), 
    State('dateout2', 'value'),
    State('monthin2', 'value'),
    State('monthout2', 'value'),
    State('yearin2', 'value'),
    State('yearout2', 'value'),
    State('symbol2', 'value'),
    State('cash', 'value'),
    State('stoploss', 'value'),
    State('batch', 'value'),
    State('portvalue', 'value')
      ])

def update_output2(n_clicks,datein2, dateout2 ,monthin2,monthout2,yearin2,yearout2,symbol2,cash,stoploss,batch,portvalue) :

    dfbacktest1 = datastocksbacktestsaham1(datein2, dateout2 ,monthin2,monthout2,yearin2,yearout2,symbol2,cash,stoploss,batch,portvalue)

    return dfbacktest1

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
