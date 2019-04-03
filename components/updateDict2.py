
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

loadModellinreg = pickle.load(open('linreg.sav', 'rb'))
loadAutoArima = pickle.load(open('autoarima.sav', 'rb'))
loadLSTM = pickle.load(open('LSTM.sav', 'rb'))

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


def updateDict3(train,test,ML,symbol) :

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
    dataset = new_data.values
    # print(new_data)
    train = new_data[:train1]
    valid = new_data[test1:]
    #     # print(new_data)
    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_test = valid.drop('Close', axis=1)
    y_test = valid['Close']

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    if ML == 'MA':
        jumlahdata = (len(new_data))
        lenvalidpreds = jumlahdata-test1

        train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()
        preds = []

        for i in range(0,lenvalidpreds):
            a = train['Close'][len(train)-lenvalidpreds+i:].sum() + sum(preds)
            b = a/lenvalidpreds
            preds.append(b)
    elif ML == 'LR':
        loadModellinreg.fit(x_train, y_train)
        preds = loadModellinreg.predict(x_test)
    elif ML == 'KNN':
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        #using gridsearch to find the best parameter
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=3)

        #fit the model and make predictions
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
    elif ML == 'AA':
        from pmdarima.arima import auto_arima
        model = auto_arima(y_train, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
        model.fit(y_train)
        # preds = loadModellinreg.predict(x_test)
        periods = len(datastockss)- len(y_train)
        preds = model.predict(n_periods=periods)
        # preds = pd.DataFrame(preds,index = valid.index,columns=['Prediction'])
    elif ML == 'LSTM':
        #importing required libraries
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # create and fit the LSTM network
        loadLSTM = Sequential()
        loadLSTM.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        loadLSTM.add(LSTM(units=50))
        loadLSTM.add(Dense(1))

        loadLSTM.compile(loss='mean_squared_error', optimizer='adam')
        loadLSTM.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        #predicting 246 values, using past 60 from the train data
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        preds = loadLSTM.predict(X_test)
        preds = scaler.inverse_transform(preds)



    datastocksML= []
    
    
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
    # print(rms) 
     
    # print(train['Close']) 

    # valid['Predictions'] = 0
    # valid['Predictions'] = preds
    # prediksiku = valid['Predictions']
    valid['Close'] = 0
    valid['Close'] = preds
    prediksiku = valid['Close']
    dfytrain = pd.DataFrame(y_train)
    dfytrain['warna']   = 'red'
    dfytrain['ML'] = ML
    dfytrain['symbol'] = symbol
    dfytrain.to_csv('dfytrain.csv')
    dfytest = pd.DataFrame(y_test)
    dfytest['warna']   = 'green'
    dfytest['ML'] = ML
    dfytest['symbol'] = symbol
    dfytest.to_csv('dfytest.csv')
    dfpred = pd.DataFrame(prediksiku)
    dfpred['warna']   = 'blue'
    dfpred['ML'] = ML
    dfpred['symbol'] = symbol
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
    dffinal.columns=['Date','Close','warna','ML','symbol']
    # dffinal['idx']=np.arange(0,len(dffinal),1)
    # dffinal['idxr']='raka'
    #=============================================
    # dffinal=dffinal.set_index('idx').reset_index()
    print(dffinal.tail())
    dffinal.to_csv('sahambaru.csv')

    return datastocksML, dffinal,rms

# updateDict3(987,987,'MA','TLKM')

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

def updateDict21(datein1, dateout1,monthin1,monthout1,yearin1,yearout1,train1,test1,symbol1):

    startyear = int(yearin1)
    startmonth = int(monthin1)
    startdate = int(datein1)
    endyear = int(yearout1)
    endmonth = int(monthout1)
    enddate = int(dateout1)
    symbols = str(symbol1)
    # datastocks = pd.DataFrame
    if symbols == str('hapusdata'):
        os.remove('saham5.csv')
        # datastocks = ('ga ada isinya')
        variable=False
    elif symbols != str(None):
        datastocks = pdr.get_data_yahoo(symbols+'.JK', start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
        variable =True
    else: 
        print('ga ada isinya!')
    
    if variable == True:
        datastocks['symbol'] = symbols
        datastocks.to_csv('saham5.csv')
        datastockss = pd.read_csv('saham5.csv')
        datastocks11 = pd.DataFrame(datastockss)
        # datastocks1 =[]
        datastocks11['symbol'] = symbols
        datastocks11.to_csv('saham5.csv')
        # datastocks1.reset_index(level=['Date'],inplace=True)
        #setting index as date values
        datastocks11['Date'] = pd.to_datetime(datastocks11.Date,format='%Y-%m-%d')
        datastocks11.index = datastocks11['Date']
        datastocks11['Datecadangan'] = datastocks11['Date']
        datastocks11['Date'] = datastocks11['Date'].map(dt.datetime.toordinal)
        # print(datastocks1)
        # print(datastocks1['Date'])
    elif variable == False:
        # datastocks.to_csv('saham4.csv')
        datastockss = pd.read_csv('saham6.csv')
        datastocks11 = pd.DataFrame(datastockss)
        datastocks11.to_csv('saham5.csv') 

    return datastocks11

def updateDict31(train1,test1,ML1,symbol1) :

    datastocks = pd.read_csv('saham5.csv')
    datastockss = pd.DataFrame(datastocks)
    # datastockss.reset_index(level=['Date'],inplace=True)
    print(datastockss)
    train1= int(train1)
    test1= int(test1)
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
    dataset = new_data.values
    # print(new_data)
    train = new_data[:train1]
    valid = new_data[test1:]
    #     # print(new_data)
    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_test = valid.drop('Close', axis=1)
    y_test = valid['Close']

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    if ML1 == 'MA':
        jumlahdata = (len(new_data))
        lenvalidpreds = jumlahdata-test1

        train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()
        preds = []

        for i in range(0,lenvalidpreds):
            a = train['Close'][len(train)-lenvalidpreds+i:].sum() + sum(preds)
            b = a/lenvalidpreds
            preds.append(b)
    elif ML1 == 'LR':
        loadModellinreg.fit(x_train, y_train)
        preds = loadModellinreg.predict(x_test)
    elif ML1 == 'KNN':
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        #using gridsearch to find the best parameter
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=3)

        #fit the model and make predictions
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
    elif ML1 == 'AA':
        loadAutoArima.fit(y_train)
        # preds = loadModellinreg.predict(x_test)
        periods = len(datastockss)- len(y_train)
        preds = model.predict(n_periods=periods)
        # preds = pd.DataFrame(preds,index = valid.index,columns=['Prediction'])
    elif ML1 == 'LSTM':
        #importing required libraries
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # create and fit the LSTM network
        loadLSTM = Sequential()
        loadLSTM.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        loadLSTM.add(LSTM(units=50))
        loadLSTM.add(Dense(1))

        loadLSTM.compile(loss='mean_squared_error', optimizer='adam')
        loadLSTM.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        #predicting 246 values, using past 60 from the train data
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

    datastocksML1= []

    
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
    # print(rms) 
     
    # print(train['Close']) 

    # valid['Predictions'] = 0
    # valid['Predictions'] = preds
    # prediksiku = valid['Predictions']
    valid['Close'] = 0
    valid['Close'] = preds
    prediksiku = valid['Close']
    dfytrain = pd.DataFrame(y_train)
    dfytrain['warna']   = 'red'
    dfytrain['ML'] = ML1
    dfytrain['symbol'] = symbol1
    dfytrain.to_csv('dfytrain.csv')
    dfytest = pd.DataFrame(y_test)
    dfytest['warna']   = 'green'
    dfytest['ML'] = ML1
    dfytest['symbol'] = symbol1
    dfytest.to_csv('dfytest.csv')
    dfpred = pd.DataFrame(prediksiku)
    dfpred['warna']   = 'blue'
    dfpred['ML'] = ML1
    dfpred['symbol'] = symbol1
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
    dffinal.columns=['Date','Close','warna','ML','symbol']
    # dffinal['idx']=np.arange(0,len(dffinal),1)
    # dffinal['idxr']='raka'
    #=============================================
    # dffinal=dffinal.set_index('idx').reset_index()
    print(dffinal.head())
    dffinal.to_csv('sahambaru1.csv')

    return datastocksML1, dffinal,rms

# updateDict31(1000,1000,'LR','isat')

def update2plotsaham(symbol1):

    saham3 = pd.read_csv('Saham3.csv')
    saham5 = pd.read_csv('Saham5.csv')

    df1 = pd.DataFrame(saham3)
    # print(df1)
    df2 = pd.DataFrame(saham5)
    # print(df2)
    dffinal = pd.DataFrame(pd.concat([df1, df2],names=['Date','Close']).reset_index())
    dffinal.columns=['no','unique','Date','High','Low','Open','Close','Volume','Adj Close','symbol']
    print(dffinal)
    dffinal.to_csv('saham2plot.csv')

    return dffinal
# update2plotsaham("TLKM","UNTR")


def update2plotsahamML(ML1,symbol1):

    sahamML = pd.read_csv('Sahambaru.csv')
    sahamML1 = pd.read_csv('Sahambaru1.csv')

    df1 = pd.DataFrame(sahamML)
    # print(df1)
    df2 = pd.DataFrame(sahamML1)
    # print(df2)
    dffinal1=[]
    dffinal1 = pd.DataFrame(pd.concat([df1, df2],names=['Date','Close']).reset_index())
    dffinal1.columns=['no','toto','Date','Close','warna','ML','symbol']
    print(dffinal1)
    dffinal1.to_csv('saham2plotML.csv')

    return dffinal1

# update2plotsahamML("raka","adli")

def datastocksbacktestsaham1(datein2, dateout2 ,monthin2,monthout2,yearin2,yearout2,symbol2,cash,stoploss,batch,portvalue):

    startyear = int(yearin2)
    startmonth = int(monthin2)
    startdate = int(datein2)
    endyear = int(yearout2)
    endmonth = int(monthout2)
    enddate = int(dateout2)
    symbols = str(symbol2)

    # startyear = 2010
    # startmonth = 1
    # startdate = 1
    # endyear = 2019
    # endmonth = 4
    # enddate = 13
    # symbols = "tlkm"

    # datastocks = web.DataReader("AAPL", "yahoo", start, end)
    datastocks = pdr.get_data_yahoo(symbols, start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
    datastocks['symbol'] = symbols
    datastocks["20d"] = np.round(datastocks["Close"].rolling(window = 20, center = False).mean(), 2)
    datastocks["50d"] = np.round(datastocks["Close"].rolling(window = 50, center = False).mean(), 2)
    datastocks["200d"] = np.round(datastocks["Close"].rolling(window = 200, center = False).mean(), 2)

    datastocks['20d-50d'] = datastocks['20d'] - datastocks['50d']

    # print(datastocks)

    datastocks["Regime"] = np.where(datastocks['20d-50d'] > 0, 1, 0)
    # We have 1's for bullish regimes and 0's for everything else. Below I replace bearish regimes's values with -1, and to maintain the rest of the vector, the second argument is datastocks["Regime"]
    datastocks["Regime"] = np.where(datastocks['20d-50d'] < 0, -1, datastocks["Regime"])

    regime_orig = datastocks.ix[-1, "Regime"]
    datastocks.ix[-1, "Regime"] = 0
    datastocks["Signal"] = np.sign(datastocks["Regime"] - datastocks["Regime"].shift(1))
    # Restore original regime data
    datastocks.ix[-1, "Regime"] = regime_orig

    datastocks_signals = pd.concat([
            pd.DataFrame({"Price": datastocks.loc[datastocks["Signal"] == 1, "Close"],
                        "Regime": datastocks.loc[datastocks["Signal"] == 1, "Regime"],
                        "Signal": "Buy"}),
            pd.DataFrame({"Price": datastocks.loc[datastocks["Signal"] == -1, "Close"],
                        "Regime": datastocks.loc[datastocks["Signal"] == -1, "Regime"],
                        "Signal": "Sell"}),
        ])
    datastocks_signals.sort_index(inplace = True)

    # print(datastocks_signals)
    # dffinal1.to_csv('saham2plotML.csv')
    datastocks_signals.to_csv('datastocks_signals.csv')


    # datastocks_signals.to_csv('datastocks_signals.csv')
    
    datastocks_signals = pd.read_csv('datastocks_signals.csv')
    datastocks_signals = pd.DataFrame(datastocks_signals)
    # print(datastocks_signals)
    #==================================
    
    if(len(datastocks_signals.loc[(datastocks_signals["Signal"].shift(1) == "Buy") & (datastocks_signals["Regime"].shift(1) == 1)].index)) == (len(datastocks_signals.loc[(datastocks_signals["Signal"] == "Buy") & datastocks_signals["Regime"] == 1, "Price"])):
        datastocks_long_profits = pd.DataFrame({
                "Price": datastocks_signals.loc[(datastocks_signals["Signal"] == "Buy") &
                                        datastocks_signals["Regime"] == 1, "Price"],
                "Profit": pd.Series(datastocks_signals["Price"] - datastocks_signals["Price"].shift(1)).loc[
                    datastocks_signals.loc[(datastocks_signals["Signal"].shift(1) == "Buy") & (datastocks_signals["Regime"].shift(1) == 1)].index
                ].tolist(),
                "End Date": datastocks_signals["Price"].loc[
                    datastocks_signals.loc[(datastocks_signals["Signal"].shift(1) == "Buy") & (datastocks_signals["Regime"].shift(1) == 1)].index
                ].index
            })
    else:    
        datastocks_backtest = pd.read_csv('datastocks_backtest0.csv')
        datastocks_backtest.to_csv('datastocks_backtest.csv')



    def ohlc_adj(dat):
        """
        :param dat: pandas DataFrame with stock data, including "Open", "High", "Low", "Close", and "Adj Close", with "Adj Close" containing adjusted closing prices
    
        :return: pandas DataFrame with adjusted stock data
    
        This function adjusts stock data for splits, dividends, etc., returning a data frame with
        "Open", "High", "Low" and "Close" columns. The input DataFrame is similar to that returned
        by pandas Yahoo! Finance API.
        """
        return pd.DataFrame({"Open": dat["Open"] * dat["Adj Close"] / dat["Close"],
                        "High": dat["High"] * dat["Adj Close"] / dat["Close"],
                        "Low": dat["Low"] * dat["Adj Close"] / dat["Close"],
                        "Close": dat["Adj Close"]})

    datastocks_adj = ohlc_adj(datastocks)
    
    # This next code repeats all the earlier analysis we did on the adjusted data
    
    datastocks_adj["20d"] = np.round(datastocks_adj["Close"].rolling(window = 20, center = False).mean(), 2)
    datastocks_adj["50d"] = np.round(datastocks_adj["Close"].rolling(window = 50, center = False).mean(), 2)
    datastocks_adj["200d"] = np.round(datastocks_adj["Close"].rolling(window = 200, center = False).mean(), 2)
    
    datastocks_adj['20d-50d'] = datastocks_adj['20d'] - datastocks_adj['50d']
    # np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not
    datastocks_adj["Regime"] = np.where(datastocks_adj['20d-50d'] > 0, 1, 0)
    # We have 1's for bullish regimes and 0's for everything else. Below I replace bearish regimes's values with -1, and to maintain the rest of the vector, the second argument is datastocks["Regime"]
    datastocks_adj["Regime"] = np.where(datastocks_adj['20d-50d'] < 0, -1, datastocks_adj["Regime"])
    # To ensure that all trades close out, I temporarily change the regime of the last row to 0
    regime_orig = datastocks_adj.ix[-1, "Regime"]
    datastocks_adj.ix[-1, "Regime"] = 0
    datastocks_adj["Signal"] = np.sign(datastocks_adj["Regime"] - datastocks_adj["Regime"].shift(1))
    # Restore original regime data
    datastocks_adj.ix[-1, "Regime"] = regime_orig

    
    # Create a DataFrame with trades, including the price at the trade and the regime under which the trade is made.
    datastocks_adj_signals = pd.concat([
            pd.DataFrame({"Price": datastocks_adj.loc[datastocks_adj["Signal"] == 1, "Close"],
                        "Regime": datastocks_adj.loc[datastocks_adj["Signal"] == 1, "Regime"],
                        "Signal": "Buy"}),
            pd.DataFrame({"Price": datastocks_adj.loc[datastocks_adj["Signal"] == -1, "Close"],
                        "Regime": datastocks_adj.loc[datastocks_adj["Signal"] == -1, "Regime"],
                        "Signal": "Sell"}),
        ])

    datastocks_adj_signals.sort_index(inplace = True)

    if(len(datastocks_adj_signals.loc[(datastocks_adj_signals["Signal"].shift(1) == "Buy") & (datastocks_adj_signals["Regime"].shift(1) == 1)].index)) == (len(datastocks_adj_signals.loc[(datastocks_adj_signals["Signal"] == "Buy") & datastocks_adj_signals["Regime"] == 1, "Price"])):
        datastocks_adj_long_profits = pd.DataFrame({
                "Price": datastocks_adj_signals.loc[(datastocks_adj_signals["Signal"] == "Buy") &
                                        datastocks_adj_signals["Regime"] == 1, "Price"],
                "Profit": pd.Series(datastocks_adj_signals["Price"] - datastocks_adj_signals["Price"].shift(1)).loc[
                    datastocks_adj_signals.loc[(datastocks_adj_signals["Signal"].shift(1) == "Buy") & (datastocks_adj_signals["Regime"].shift(1) == 1)].index
                ].tolist(),
                "End Date": datastocks_adj_signals["Price"].loc[
                    datastocks_adj_signals.loc[(datastocks_adj_signals["Signal"].shift(1) == "Buy") & (datastocks_adj_signals["Regime"].shift(1) == 1)].index
                ].index
            })
        datastocks_adj_long_profits1 = True
    else:    
        datastocks_backtest = pd.read_csv('datastocks_backtest0.csv')
        datastocks_backtest.to_csv('datastocks_backtest.csv')
        datastocks_adj_long_profits1 = False

    # print(datastocks_adj_long_profits)
    
    datastocksbacktestsaham=[]

    # We need to get the low of the price during each trade.
    if datastocks_adj_long_profits1 == False :
        datastocks_backtest = pd.read_csv('datastocks_backtest0.csv')
        datastocks_backtest.to_csv('datastocks_backtest.csv')
    else:
        tradeperiods = pd.DataFrame({"Start": datastocks_adj_long_profits.index,
                                    "End": datastocks_adj_long_profits["End Date"]})
        datastocks_adj_long_profits["Low"] = tradeperiods.apply(lambda x: min(datastocks_adj.loc[x["Start"]:x["End"], "Low"]), axis = 1)
    # print(datastocks_adj_long_profits)

    # Now we have all the information needed to simulate this strategy in datastocks_adj_long_profits
    if datastocks_adj_long_profits1 == True:
        cash = 100000
        datastocks_backtest = pd.DataFrame({"Start Port. Value": [],
                                "End Port. Value": [],
                                "End Date": [],
                                "Shares": [],
                                "Share Price": [],
                                "Trade Value": [],
                                "Profit per Share": [],
                                "Total Profit": [],
                                "Stop-Loss Triggered": []})
        port_value = .1  # Max proportion of portfolio bet on any trade
        batch = 100      # Number of shares bought per batch
        stoploss = .1    # % of trade loss that would trigger a stoploss
        for index, row in datastocks_adj_long_profits.iterrows():
            batches = np.floor(cash * port_value) // np.ceil(batch * row["Price"]) # Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Price"] # How much money is put on the line with each trade
            if row["Low"] < (1 - stoploss) * row["Price"]:   # Account for the stop-loss
                share_profit = np.round((1 - stoploss) * row["Price"], 2)
                stop_trig = True
            else:
                share_profit = row["Profit"]
                stop_trig = False
            profit = share_profit * batches * batch # Compute profits
            # Add a row to the backtest data frame containing the results of the trade
            datastocks_backtest = datastocks_backtest.append(pd.DataFrame({
                        "Start Port. Value": cash,
                        "End Port. Value": cash + profit,
                        "End Date": row["End Date"],
                        "Shares": batch * batches,
                        "Share Price": row["Price"],
                        "Trade Value": trade_val,
                        "Profit per Share": share_profit,
                        "Total Profit": profit,
                        "Stop-Loss Triggered": stop_trig
                    }, index = [index]))
            cash = max(0, cash + profit)
        
        print(datastocks_backtest)
        datastocks_backtest.to_csv('datastocks_backtest.csv')

        # print(datastocks_backtest["End Port. Value"])

        # plt.plot(datastocks_backtest["End Port. Value"])
        # plt.show()
    else:
        datastocks_backtest = pd.read_csv('datastocks_backtest0.csv')
        datastocks_backtest.to_csv('datastocks_backtest.csv')
    dfbacktest1 = []
    
    return datastocksbacktestsaham,dfbacktest1

# datastocksbacktestsaham1('')
# datastocksbacktestsaham1(12, 12 ,12,12,2000,2100,'ANTM')