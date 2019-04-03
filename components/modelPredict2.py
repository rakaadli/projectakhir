
import dash_core_components as dcc
import dash_html_components as html

def renderModelPredict() :
    return html.Div([
                html.H1('Test Saham', className='h1'),
                 html.Div(children=[
                    html.Div([
                        html.P('Tanggal mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='datein', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tanggal akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='dateout', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Bulan mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthin', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Bulan akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthout', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearin', type='number', value='2000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearout', type='number', value='2100')
                    ],className='col-4'),
                    html.Div([
                        html.P('Data Train : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='train', type='number', value='1000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Data Test : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='test', type='number', value='1000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Symbol saham : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='symbol', type='text')
                    ],className='col-4'),
                    html.Div([
                        html.P('Jenis Machine Learning : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Dropdown(id='ML', options=[
                            {'label' : 'Moving Averange', 'value' : 'MA'},
                            {'label' : 'Linear Regresression', 'value' : 'LR'},
                            {'label' : 'On Proggress', 'value' : 'KNN'},
                            {'label' : 'On Proggress', 'value' : 'AA'},
                            {'label' : 'On Proggress', 'value' : 'LSTM'}
                        ], value='None.1' )
                    ],className='col-4'),
                    html.Div([
                        html.A("Daftar Kode Saham", href='https://www.idx.co.id/data-pasar/data-saham/daftar-saham/', target="_blank")
                    ],className='col-2'),
                    html.Div([
                        html.Button('Predict', type='submit', id='buttonPredict', className='btn btn-primary')
                        # html.A(
                        #     html.Button('Predict', id='buttonPredict', className='btn btn-primary'),
                        #     href = 'http://127.0.0.1:2019/'
                        # )
                    ],className='mx-auto', style={ 'paddingTop': '20px', 'paddingBottom': '20px' })
                ],className='row'),
                html.Div([
                    html.H2('', id='outputPredict', className='mx-auto')
                ], className='row')
            ])
                    

def renderModelPredict1() :
    return html.Div([
                html.H1('Test Saham 2', className='h1'),
                 html.Div(children=[
                    html.Div([
                        html.P('Tanggal mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='datein1', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tanggal akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='dateout1', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Bulan mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthin1', type='number', value='12')
                    ],className='col-4'),
                    
                    html.Div([
                        html.P('Bulan akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthout1', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearin1', type='number', value='2000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearout1', type='number', value='2100')
                    ],className='col-4'),
                    html.Div([
                        html.P('Data Train : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='train1', type='number', value='1000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Data Test : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='test1', type='number', value='1000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Symbol saham : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='symbol1', type='text')
                    ],className='col-4'),
                    html.Div([
                        html.P('Jenis Machine Learning : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Dropdown(id='ML1', options=[
                            {'label' : 'Moving Averange', 'value' : 'MA'},
                            {'label' : 'Linear Regresression', 'value' : 'LR'},
                            {'label' : 'On Proggress', 'value' : 'KNN'},
                            {'label' : 'On Proggress', 'value' : 'AA'},
                            {'label' : 'On Proggress', 'value' : 'LSTM'}
                        ], value='None.1' )
                    ],className='col-4'),
                    html.Div([
                        html.Button('Predict', type='submit', id='buttonPredict1', className='btn btn-primary')
                        # html.A(
                        #     html.Button('Predict', id='buttonPredict', className='btn btn-primary'),
                        #     href = 'http://127.0.0.1:2019/'
                        # )
                    ],className='mx-auto', style={ 'paddingTop': '20px', 'paddingBottom': '20px' })
                ],className='row'),
                html.Div([
                    html.H2('', id='outputPredict1', className='mx-auto')
                ], className='row')
            ])
                  
def renderModelPredict2() :
    return html.Div([
                html.H1('Test Saham Trading Signal', className='h1'),
                 html.Div(children=[
                    html.Div([
                        html.P('Tanggal mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='datein2', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tanggal akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='dateout2', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Bulan mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthin2', type='number', value='12')
                    ],className='col-4'),
                    
                    html.Div([
                        html.P('Bulan akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='monthout2', type='number', value='12')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun mulai prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearin2', type='number', value='2000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Tahun akhir prediksi : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='yearout2', type='number', value='2100')
                    ],className='col-4'),
                    html.Div([
                        html.P('Symbol saham : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='symbol2', type='text')
                    ],className='col-4'),
                    html.Div([
                        html.P('Cash (0-100000): ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='cash', type='number', value='100000')
                    ],className='col-4'),
                    html.Div([
                        html.P('Stoploss (0.1-0.9) : ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='stoploss', type='number', value='1000')
                    ],className='col-4'),
                    html.Div([
                        html.P('batch (0-10000): ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='batch', type='text')
                    ],className='col-4'),
                    html.Div([
                        html.P('Port Value (0.1-0.9): ')
                     ],className='col-2'),
                    html.Div([
                        dcc.Input(id='portvalue', type='text')
                    ],className='col-4'),
                    html.Div([
                        html.Button('Predict', type='submit', id='buttonPredict2', className='btn btn-primary')
                        # html.A(
                        #     html.Button('Predict', id='buttonPredict', className='btn btn-primary'),
                        #     href = 'http://127.0.0.1:2019/'
                        # )
                    ],className='mx-auto', style={ 'paddingTop': '20px', 'paddingBottom': '20px' })
                ],className='row'),
                html.Div([
                    html.H2('', id='outputPredict2', className='mx-auto')
                ], className='row')
            ])
                  