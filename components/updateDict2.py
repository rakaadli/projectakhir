import pandas_datareader as pdr
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt

def updateDict2(datein, dateout ,monthin,monthout,yearin,yearout,train,test,symbol) :

    startyear = int(yearin)
    startmonth = int(monthin)
    startdate = int(datein)
    endyear = int(yearout)
    endmonth = int(monthout)
    enddate = int(dateout)
    symbols = str(symbol)
    datastocks = pd.DataFrame
    if symbols != str(None):
        datastocks = pdr.get_data_yahoo(symbols+'.JK', start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
    else: 
        print('ga ada isinya!')
    
    datastocks.to_csv('saham.csv')
    datastockss = pd.read_csv('saham.csv')
    datastocks1 = pd.DataFrame(datastockss)
    datastocks1.to_csv('saham2.csv')
    # datastocks1.reset_index(level=['Date'],inplace=True)
    #setting index as date values
    datastocks1['Date'] = pd.to_datetime(datastocks1.Date,format='%Y-%m-%d')
    datastocks1.index = datastocks1['Date']
    datastocks1['Datecadangan'] = datastocks1['Date']
    datastocks1['Date'] = datastocks1['Date'].map(dt.datetime.toordinal)
    print(datastocks1)
    print(datastocks1['Date'])

    return datastocks1

# updateDict2(10, 10 ,10,10,2000,2020,100,100,'TLKM')