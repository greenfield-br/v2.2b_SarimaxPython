from func_base import list_Code
import os.path
import datetime
import pandas
import urllib.request
from urllib.error import URLError, HTTPError
import json

frequency = list_Code('../etc', 'frequency_code.txt')
symbols = list_Code('../etc', 'Mrcd_code.txt')
folder = '../raw_data'
folder_source = 'mrcd'
strColumns = ['date', 'opening', 'closing', 'lowest', 'highest', 'volume', 'quantity', 'amount', 'avg_price']

for i in symbols:
   #determines yesterday as datetime object
   yesterday = datetime.datetime.today() - datetime.timedelta(0)
   #set yesterday's reference time to 00:00:00
   yesterday = datetime.datetime(yesterday.year, yesterday.month, yesterday.day)

   filename = 'mrcd_' + i + '.csv'
   if os.path.isfile(folder + '/' + folder_source + '/' + filename) == False:
      print('# no exisiting raw data file.')
      #if no previous data was retrieved, start from 01/01/2013.
      fetch_day = datetime.datetime(2013, 6, 20)
      with open(folder + '/' + folder_source + '/' + filename, "w") as outfile:
        for x in strColumns:
            outfile.write(x)
            if x == 'avg_price':
               outfile.write('\n')
            else:
               outfile.write(',')
   else:
      #reads data file as dataframe.
      a = pandas.read_csv(folder + '/' + folder_source + '/' + filename)
      a = a.set_index('date', drop = True)
      #gets latest datetime
      latest_datetime_exfile = a.index[-1]
      latest_datetime = a.index[-1]
      latest_datetime = pandas.to_datetime(latest_datetime)
      fetch_day = latest_datetime + datetime.timedelta(1)

   url_base = 'https://www.mercadobitcoin.net/api/' + i + '/day-summary/'
#   while fetch_day < yesterday:
   while fetch_day < yesterday:
      url = url_base + fetch_day.strftime('%Y/%m/%d')

      with urllib.request.urlopen(url) as x:
         response = urllib.request.urlopen(urllib.request.Request(url))
         a = json.loads(x.read().decode())
      a = pandas.DataFrame(a, index=['date'])
      if a.empty == True:
         break
      a = a.set_index('date', drop = True)
      a.to_csv(folder + '/' + folder_source + '/' + filename, mode = 'a', header = None)
      fetch_day += datetime.timedelta(1)
   print('0k')

for i in symbols:
   filename = 'mrcd_' + i + '.csv'
   a = pandas.read_csv(folder + '/' + folder_source + '/' + filename)
   a = a.set_index('date', drop = True)
   strColumns = ['opening', 'highest', 'lowest', 'closing', 'volume', 'quantity', 'amount', 'avg_price']
   a = a[strColumns]
   filename = 'mrcd-' + i + 'BRL_xD.csv'
   a.to_csv(folder + '/' + filename, mode = 'w', header = strColumns)
print('1k')

