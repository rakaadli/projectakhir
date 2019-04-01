<<<<<<< HEAD
# import csv
# import urllib.request
# # import request
# url = 'https://query1.finance.yahoo.com/v7/finance/download/TLKM.JK?period1=1096304400&period2=1551459600&interval=1d&events=history&crumb=IQQexnVXl2p'
# csv = urllib.request.urlopen(url).read() # returns type 'str'
# with open('filesaham.csv', 'wb') as fx: # str, hence mode 'w'
#     fx.write(csv)

# import requests
# import shutil
# def callme():
#     url = "http://real-chart.finance.yahoo.com/table.csv?s=%5EBSESN&a=03&b=3&c=1997&d=10&e=4&f=2015&g=d&ignore=.csv"
#     r = requests.get(url, verify=False,stream=True)
#     if r.status_code!=200:
#         print("Failure!!")
#         exit()
#     else:
#         r.raw.decode_content = True
#         with open("file1.csv", 'wb') as f:
#             shutil.copyfileobj(r.raw, f)
#         print("Success")

# if __name__ == '__main__':
#     callme()

# import urllib.request
# with urllib.request.urlopen('http://python.org/') as response:
#    html = response.read()

# from yahoo_finance import Share
# yahoo = Share('YHOO')
# print(yahoo)

# import yahoo_finance
# import pandas as pd

# symbol = yahoo_finance.Share("GOOG")
# google_data = symbol.get_historical("1999-01-01", "2016-06-30")
# google_df = pd.DataFrame(google_data)

# # Output data into CSV
# google_df.to_csv("/home/username/google_stock_data.csv")

import pandas_datareader as pdr
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# startyear =  int(input('ketikan tahun awal mulai:'))
# startmonth = int(input('ketikan bulan awal mulai:'))
# startdate = int(input('ketikan date awal mulai:'))

# endyear = int(input('ketikan tahun berakhir mulai: '))
# endmonth = int(input('ketikan bulan berakhir mulai: '))
# enddate = int(input('ketikan date berakhir mulai: '))

# symbols = input('ketikan symbol saham: ')
symbol = "TLKM"
startyear = 2010
startmonth = 1
startdate = 1
endyear = 2019
endmonth = 4
enddate = 13
symbols = symbol



datastocks = pdr.get_data_yahoo(symbols+".JK", start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
# print(datastocks.info())
# print(datastocks)


# plt.plot(datastocks['Close'], 'r-')
# plt.show()

# print(datastocks['Adj Close'])

# datastocks.to_csv('saham.csv')


=======
# import csv
# import urllib.request
# # import request
# url = 'https://query1.finance.yahoo.com/v7/finance/download/TLKM.JK?period1=1096304400&period2=1551459600&interval=1d&events=history&crumb=IQQexnVXl2p'
# csv = urllib.request.urlopen(url).read() # returns type 'str'
# with open('filesaham.csv', 'wb') as fx: # str, hence mode 'w'
#     fx.write(csv)

# import requests
# import shutil
# def callme():
#     url = "http://real-chart.finance.yahoo.com/table.csv?s=%5EBSESN&a=03&b=3&c=1997&d=10&e=4&f=2015&g=d&ignore=.csv"
#     r = requests.get(url, verify=False,stream=True)
#     if r.status_code!=200:
#         print("Failure!!")
#         exit()
#     else:
#         r.raw.decode_content = True
#         with open("file1.csv", 'wb') as f:
#             shutil.copyfileobj(r.raw, f)
#         print("Success")

# if __name__ == '__main__':
#     callme()

# import urllib.request
# with urllib.request.urlopen('http://python.org/') as response:
#    html = response.read()

# from yahoo_finance import Share
# yahoo = Share('YHOO')
# print(yahoo)

# import yahoo_finance
# import pandas as pd

# symbol = yahoo_finance.Share("GOOG")
# google_data = symbol.get_historical("1999-01-01", "2016-06-30")
# google_df = pd.DataFrame(google_data)

# # Output data into CSV
# google_df.to_csv("/home/username/google_stock_data.csv")

import pandas_datareader as pdr
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# startyear =  int(input('ketikan tahun awal mulai:'))
# startmonth = int(input('ketikan bulan awal mulai:'))
# startdate = int(input('ketikan date awal mulai:'))

# endyear = int(input('ketikan tahun berakhir mulai: '))
# endmonth = int(input('ketikan bulan berakhir mulai: '))
# enddate = int(input('ketikan date berakhir mulai: '))

# symbols = input('ketikan symbol saham: ')
symbol = "TLKM"
startyear = 2010
startmonth = 1
startdate = 1
endyear = 2019
endmonth = 4
enddate = 13
symbols = symbol



datastocks = pdr.get_data_yahoo(symbols+".JK", start=datetime(startyear, startmonth, startdate), end=datetime(endyear, endmonth, enddate))
# print(datastocks.info())
# print(datastocks)


# plt.plot(datastocks['Close'], 'r-')
# plt.show()

# print(datastocks['Adj Close'])

# datastocks.to_csv('saham.csv')


>>>>>>> d2791a0e578a74cec91e1c325be0f51b9fe0268b
print(symbols)