import dash_html_components as html
import dash_table as dt
import pandas as pd
import numpy as np
import dash_table


PAGE_SIZE1 = 10
PAGE_SIZE2 = 10
PAGE_SIZE3 = 10

saham = pd.read_csv('saham.csv')


def generate_table(dataframe) :
    return dt.DataTable(
        id='table-multicol-sorting',
        columns=[
            {"name": i, "id": i} for i in dataframe.columns
        ],
        pagination_settings={
            'current_page': 0,
            'page_size': PAGE_SIZE1
        },
        pagination_mode='be',

        sorting='be',
        sorting_type='multi',
        sorting_settings=[],
        style_table={'overflowX': 'scroll'}
    )

def generate_table1(dataframe) :
    return dt.DataTable(
        id='table-multicol-sorting1',
        columns=[
            {"name": x, "id": x} for x in dataframe.columns
        ],
        pagination_settings={
            'current_page': 0,
            'page_size': PAGE_SIZE2
        },
        pagination_mode='be',

        sorting='be',
        sorting_type='multi',
        sorting_settings=[],
        style_table={'overflowX': 'scroll'}
    )


def renderTable(df1) :
    return html.Div([
                html.H1('Tabel Saham 1', className='h1'),
                generate_table(df1),
                # html.H1('Tabel Saham 2', className='h1'),
                # generate_table1(df2)
            ])
# def generate_table1(dataframe) :
#     return dt.DataTable(
#         id='table-multicol-sorting1',
#         columns=[
#             {"name": x, "id": x} for x in dataframe.columns
#         ],
#         pagination_settings={
#             'current_page': 0,
#             'page_size': PAGE_SIZE2
#         },
#         pagination_mode='be',

#         sorting='be',
#         sorting_type='multi',
#         sorting_settings=[],
#         style_table={'overflowX': 'scroll'}
#     )

# def renderTable1(df) :
#     return html.Div([
#                 html.H1('Tabel Saham 2', className='h1'),
#                 generate_table1(df)

#             ])

# def generate_table2(dataframe) :
#     return dt.DataTable(
#         id='table-multicol-sorting2',
#         columns=[
#             {"name": x, "id": x} for x in dataframe.columns
#         ],
#         pagination_settings={
#             'current_page': 0,
#             'page_size': PAGE_SIZE3
#         },
#         pagination_mode='be',

#         sorting='be',
#         sorting_type='multi',
#         sorting_settings=[],
#         style_table={'overflowX': 'scroll'}
#     )

# def renderTable2(df) :
#     return html.Div([
#                 html.H1('Tabel Saham 3', className='h1'),
#                 generate_table2(df)

#             ])
dfsignal = pd.read_csv('datastocks_signals.csv')
dfbacktest = pd.read_csv('datastocks_backtest.csv')
def renderTablesignal1() :
    return  html.Div([
            html.H4('Tabel Signal Saham 1', className='h1'),
            dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in dfsignal.columns],
            data=dfsignal.to_dict("rows"),
            style_table={
            'maxHeight': '300',
            'overflowY': 'scroll'
              }
            )
             ])

def renderTablebacktest1() :
    return  html.Div([
            html.H4('Tabel Backtest Saham 1', className='h1'),
            dash_table.DataTable(
            id='table1',
            columns=[{"name": i, "id": i} for i in dfbacktest.columns],
            data=dfbacktest.to_dict("rows"),
            style_table={
            'maxHeight': '300',
            'overflowY': 'scroll'
              }
            )
             ])