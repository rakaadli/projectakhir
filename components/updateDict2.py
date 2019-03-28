import pandas_datareader as pdr
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

loadModellogreg = pickle.load(open('logreg.sav', 'rb'))

def updateDict2(datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol):

    startyear = int(yearin)
    startmonth = int(monthin)
    startdate = int(datein)
    endyear = int(yearout)
    endmonth = int(monthout)
    enddate = int(dateout)
    symbols = str(symbol)
    # datastocks = pd.DataFrame
    if symbols == str('hapusdata'):
        os.remove('saham3.csv')
        # datastocks = ('ga ada isinya')
        variable=False
    elif symbols != str(None):
        datastocks = pdr.get_data_yahoo(symbols+'.JK', start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
        variable =True
    else: 
        print('ga ada isinya!')
    
    if variable == True:
        datastocks['symbol'] = symbols
        datastocks.to_csv('saham3.csv')
        datastockss = pd.read_csv('saham3.csv')
        datastocks1 = pd.DataFrame(datastockss)
        # datastocks1 =[]
        datastocks1['symbol'] = symbols
        datastocks1.to_csv('saham3.csv')
        # datastocks1.reset_index(level=['Date'],inplace=True)
        #setting index as date values
        datastocks1['Date'] = pd.to_datetime(datastocks1.Date,format='%Y-%m-%d')
        datastocks1.index = datastocks1['Date']
        datastocks1['Datecadangan'] = datastocks1['Date']
        datastocks1['Date'] = datastocks1['Date'].map(dt.datetime.toordinal)
        print(datastocks1)
        # print(datastocks1['Date'])
    elif variable == False:
        # datastocks.to_csv('saham4.csv')
        datastockss = pd.read_csv('saham4.csv')
        datastocks1 = pd.DataFrame(datastockss)
        datastocks1.to_csv('saham3.csv') 

    return datastocks1

# updateDict2(10, 10 ,10,10,2000,2020,100,100,'TLKM')

#=================================

# def updateDict2(datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol):

#     startyear = int(yearin)
#     startmonth = int(monthin)
#     startdate = int(datein)
#     endyear = int(yearout)
#     endmonth = int(monthout)
#     enddate = int(dateout)
#     symbols = str(symbol)
#     # datastocks = pd.DataFrame
#     if symbols != str(None):
#         datastocks = pdr.get_data_yahoo(symbols+'.JK', start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))


#     else: 
#         print('ga ada isinya!')
    
#     datastocks.to_csv('saham.csv')
#     datastockss = pd.read_csv('saham.csv')
#     datastocks1 = pd.DataFrame(datastockss)
#     # datastocks1 =[]
#     datastocks1.to_csv('saham2.csv')
#     # datastocks1.reset_index(level=['Date'],inplace=True)
#     #setting index as date values
#     datastocks1['Date'] = pd.to_datetime(datastocks1.Date,format='%Y-%m-%d')
#     datastocks1.index = datastocks1['Date']
#     datastocks1['Datecadangan'] = datastocks1['Date']
#     datastocks1['Date'] = datastocks1['Date'].map(dt.datetime.toordinal)
#     print(datastocks1)
#     # print(datastocks1['Date'])

#     return datastocks1

# updateDict2(10, 10 ,10,10,2000,2020,100,100,'UNVR')

def updateDict3(train,test) :

    datastocks = pd.read_csv('saham3.csv')
    datastockss = pd.DataFrame(datastocks)
    # datastockss.reset_index(level=['Date'],inplace=True)
    print(datastockss)
    train1= int(train)
    test1= int(test)
    # print(train)
    # print(test)

    #setting index as date values
    datastockss['Date'] = pd.to_datetime(datastockss.Date,format='%Y-%m-%d')
    datastockss.index = datastockss['Date']
    datastockss['Datecadangan'] = datastockss['Date']
    datastockss['Date'] = datastockss['Date'].map(dt.datetime.toordinal)
    data = datastockss.sort_index(ascending=True, axis=0)
    # print(datastockss)

    # #creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(datastockss)),columns=['Date', 'Close'])

    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    new_data.index = datastockss['Datecadangan']
    # print(new_data)
    train = new_data[:train1]
    valid = new_data[test1:]
    #     # print(new_data)
    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_test = valid.drop('Close', axis=1)
    y_test = valid['Close']


    jumlahdata = (len(new_data))
    lenvalidpreds = jumlahdata-test1

    # train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()
    # preds = []

    # for i in range(0,lenvalidpreds):
    #     a = train['Close'][len(train)-lenvalidpreds+i:].sum() + sum(preds)
    #     b = a/lenvalidpreds
    #     preds.append(b)
   

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    loadModellogreg.fit(x_train, y_train)
    datastockslogreg= []

    preds = loadModellogreg.predict(x_test)
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
    print(rms) 
     
    # print(train['Close']) 

    # valid['Predictions'] = 0
    # valid['Predictions'] = preds
    # prediksiku = valid['Predictions']
    valid['Close'] = 0
    valid['Close'] = preds
    prediksiku = valid['Close']
    dfytrain = pd.DataFrame(y_train)
    dfytrain['warna']   = 'red'
    dfytrain.to_csv('dfytrain.csv')
    dfytest = pd.DataFrame(y_test)
    dfytest['warna']   = 'green'
    dfytest.to_csv('dfytest.csv')
    dfpred = pd.DataFrame(prediksiku)
    dfpred['warna']   = 'blue'
    print(dfpred)
    #================================
    # new_data.index = datastockss['Datecadangan']
    # valid.index = new_data[test1:].index  
    # train.index = new_data[:train1].index  

    # print(preds)
    # print(new_data)

    # print(new_data)
    # print(train['Close']) 
    # print(valid['Close'])
    # print(valid['Predictions'])

    # #plot
    # plt.plot(train['Close'],color='r')
    # plt.plot(valid['Close'],color='b')
    # plt.plot(valid['Predictions'],color='g')
    # plt.show()

    #============================================
    dffinal = pd.DataFrame(pd.concat([dfytrain, dfytest,dfpred],names=['Date','Close']).reset_index())
    dffinal.columns=['Date','Close','warna']
    # dffinal['idx']=np.arange(0,len(dffinal),1)
    # dffinal['idxr']='raka'
    #=============================================
    # dffinal=dffinal.set_index('idx').reset_index()
    print(dffinal.tail())
    dffinal.to_csv('sahambaru.csv')

    return datastockslogreg, dffinal

# updateDict3(987,987)

# dffinal = pd.read_csv('sahambaru.csv')
# dfytrain = pd.read_csv('dfytrain.csv')
# dfytest = pd.read_csv('dfytest.csv')
# dfypred = pd.read_csv('dfypred.csv')

# app = dash.Dash(__name__)

# server = app.server

# app.layout = html.Div([
#                 html.H1('Plot Saham', className='h1'),
#                 dcc.Graph(
#                     id='Plot1',
#                     figure={
#                         'data': [
#                             go.Scatter(
#                                 # x=dfytrain['Datecadangan'],
#                                 # y=dfytrain['Close'],
#                                 x=dffinal['Date'],
#                                 y=dffinal['Close'],
#                                 mode='lines',
#                                 marker=dict(color='rgb(205, 12, 24)', size=10, line=dict(width=0.5, color='white'))
#                                 # name=legend(col)
#                             ) 
#                             # for col in df['Type'].unique()
#                         ],
#                         'layout': go.Layout(
#                             xaxis= dict(title='Actual Harga saham'),
#                             yaxis={'title': 'Price'},
#                             margin={ 'l': 40, 'b': 40, 't': 10, 'r': 10 },
#                             hovermode='closest'
#                         )
#                     }
#                 )
#             ])

# if __name__ == '__main__':
#     # app.run_server(debug=True, port=2019)
#     app.run_server(debug=True, port=2021)
