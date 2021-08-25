#import bitmex
#client = bitmex.bitmex(api_key='jCk7Hxn-UwDnH811_J1g59FU', api_secret='p6O7MhhVHlMFIGWkYZ-0EpmL-8ulW7flcKIw_RAYQZfrSUP8')
#client.Quote.Quote_get(symbol='XBTUSD').result()

# bitmex sends eod candle of day X as day X+1

from func_base import list_Code
import os.path
import requests
import json
import pandas
import datetime
import shutil

frequency = list_Code('../etc', 'frequency_code.txt')
symbols = list_Code('../etc', 'Bitmex_code.txt')
folder = '../raw_data'
folder_source = 'bitmex'
for i in symbols:
   filename = 'bitmex_' + i + '.csv'
   url_base = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1d&partial=true&symbol=' + i + 'USD&reverse=false&startTime='
   strColumns = ['open', 'high', 'low', 'close', 'volume', 'trades']

#   if os.path.isfile(folder + '/' + folder_source + '/' + filename) == False:
#   print('no')
   url = url_base + '2013-01-01&endTime=2018-11-25'
   request_get = requests.get(url)
   a = pandas.read_json(request_get.content)
   a = a.set_index('timestamp', drop = True)
   a = a[strColumns]
   a.to_csv(folder + '/' + folder_source + '/' + filename, mode = 'w')
   del a
   #determines yesterday as datetime object
   yesterday = datetime.datetime.today() - datetime.timedelta(1)
   #set yesterday's reference time to 23:59:59
   yesterday = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
   #reads data file as dataframe.
   a = pandas.read_csv(folder + '/' + folder_source + '/' + filename)
   #gets latest datetime
   latest_datetime = a.iloc[-1,0]
   latest_datetime = pandas.to_datetime(latest_datetime)
   latest_datetime = datetime.datetime(latest_datetime.year, latest_datetime.month, latest_datetime.day, 23, 59, 59)
   #loops until latest_datetime becomes yesterday.
   batch_size_max = 100
   while latest_datetime < yesterday:
      start_date = (latest_datetime + datetime.timedelta(1)).strftime('%Y-%m-%d')
      batch_size = min((yesterday - latest_datetime).days, batch_size_max)
      end_date = (latest_datetime + datetime.timedelta(1 + batch_size)).strftime('%Y-%m-%d')
      url = url_base + start_date + '&endTime=' + end_date
      request_get = requests.get(url)
      a = pandas.read_json(request_get.content)
      if a.empty == True:
         break
      a = a.set_index('timestamp', drop = True)
      x = pandas.to_datetime(a.index)
      x = x - datetime.timedelta(1)
      a = a.set_index(x, drop = True)
      a = a[strColumns]
      a.to_csv(folder + '/' + folder_source + '/' + filename, mode = 'a', header = False)
      latest_datetime = a.index[-1] + datetime.timedelta(1)
      latest_datetime = pandas.to_datetime(latest_datetime)
      print(latest_datetime)
   print('0k')
#   a = pandas.read_csv(folder + '/' + folder_source + '/' + filename)
#   x = pandas.to_datetime(a.iloc[:,0])
#   x = x - datetime.timedelta(1)
#   a = a.set_index(x, drop = True)
#   a = a[strColumns]
#   a.to_csv(folder + '/' + folder_source + '/' + filename, mode = 'w', header = True)
   del a
   source_file = folder + '/' + folder_source + '/' + filename
   filename = 'bitmex-' + i + 'USD_xD.csv'
   destination_file = folder + '/' + '/' + filename
   shutil.copy2(source_file, destination_file)
print('1k')
