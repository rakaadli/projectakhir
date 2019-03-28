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
                        dcc.Dropdown(id='inputML', options=[
                            {'label' : 'Moving Averange', 'value' : 'MA'},
                            {'label' : 'Linear Regresression', 'value' : 'LR'}
                        ], value='None.1' )
                    ],className='col-4'),
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
                    




            