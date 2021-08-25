import os
import re
import pandas
import numpy
from datetime import datetime
from func_base import list_Code

str2date = lambda x : datetime.strptime(x,'%d.%m.%Y %H:%M:%S.%f').strftime('%Y-%m-%d')

frequency = list_Code('./etc', 'frequency_code.txt')
symbols = list_Code('./etc', 'Coinmarketcap_code.txt')
folder = './raw_data'
folder_source = '/dukascopy'

for i in os.listdir(folder +folder_source +'/'):
   if i.endswith("dukascopy.csv"):
      sfile = re.split('-|x|_',i)
      l1 = sfile[0]
      l2 = sfile[1]
      f  = sfile[2]
      print(l1, l2, f)
      #tests it it's a forex pair. to be reversed later.
      if (l2 != ''):
         filename = folder + folder_source +'/' +l1 +'-' +l2 +'x' +f +'_dukascopy.csv'
      #else it is a non forex asset priced in USD. won't be reversed.
      else:
         filename = folder + folder_source +'/' +l1 +'_x' +f +'_dukascopy.csv'
      a = pandas.read_csv(filename, index_col=0, date_parser=str2date)
      col_name = ["Gmt time","Open","High","Low","Close","Volume"]
      for j in frequency:
         a0 = a.groupby(pandas.TimeGrouper(j))[col_name[1]].first()
         a1 = a.groupby(pandas.TimeGrouper(j))[col_name[2]].max()
         a2 = a.groupby(pandas.TimeGrouper(j))[col_name[3]].min()
         a3 = a.groupby(pandas.TimeGrouper(j))[col_name[4]].last()
         a4 = a.groupby(pandas.TimeGrouper(j))[col_name[5]].sum()
         b = [a0, a1, a2, a3, a4]
         b = pandas.concat(b, axis=1)
         b.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
         if (l2 != ''):
            filename = folder +'/' +l1 +'_' +l2 +'x' +j +'.csv'
         else:
            filename = folder +'/' +l1 +'_' +'x' +j +'.csv'
         b = b[b.index < datetime(datetime.now().year, datetime.now().month, datetime.now().day)]
         b.to_csv(filename, date_format='%Y-%m-%d')
         #reverse forex pair
         if (l2 != ''):
            b = [1/a0, 1/a2, 1/a1, 1/a3, a4]
            b = pandas.concat(b, axis=1)
            b.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            filename = folder +'/' +l2 +'_' +l1 +'x' +j +'.csv'
            b = b[b.index < datetime(datetime.now().year, datetime.now().month, datetime.now().day)]
            b.to_csv(filename, date_format='%Y-%m-%d')
