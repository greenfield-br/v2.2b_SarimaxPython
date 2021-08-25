import pandas
import warnings
from datetime import datetime
from func_base import list_Code
from cryptocmd import CmcScraper

warnings.filterwarnings('ignore')
frequency = list_Code('../etc', 'frequency_code.txt')
symbols = list_Code('../etc', 'Coinmarketcap_code.txt')
folder = '../raw_data'
folder_source = 'coinmarketcap'

for i in symbols:
   date_end = datetime.now() #+ timedelta(days=1)
   date_end = date_end.strftime('%d-%m-%Y')
   scraper = CmcScraper(i, '27-12-2013', date_end)
   headers, data = scraper.get_data()
   filename = '{}/{}/{}_data.csv'.format(folder, folder_source, i)
   scraper.export_csv(filename)
# Pandas dataFrame for the same data
   a = scraper.get_dataframe()
   a = a.set_index(pandas.DatetimeIndex(a['Date']))
   for j in frequency: #[0,1,2,3,4]:
      a0 = a.groupby(pandas.TimeGrouper(j))[headers[1]].first()
      a1 = a.groupby(pandas.TimeGrouper(j))[headers[2]].max()
      a2 = a.groupby(pandas.TimeGrouper(j))[headers[3]].min()
      a3 = a.groupby(pandas.TimeGrouper(j))[headers[4]].last()
      a4 = a.groupby(pandas.TimeGrouper(j))[headers[5]].sum()
      b = [a0, a1, a2, a3, a4]
      b = pandas.concat(b, axis=1)
      b.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#      b = b[b.index < datetime(datetime.now().year, datetime.now().month, datetime.now().day)]
      b.to_csv("{}/{}USD_x{}.csv".format(folder, i, j), date_format='%Y-%m-%d')
