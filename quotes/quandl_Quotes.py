import pandas
import quandl
quandl.ApiConfig.api_key = "8jRUz3xWAcL1eM3-d9TN"
import warnings
from datetime import datetime
from func_base import list_Code

warnings.filterwarnings('ignore')
frequency = list_Code('../etc', 'frequency_code.txt')
folder = '../raw_data'
folder_source = 'quandl'

symbols = ['BCHARTS/coinbaseUSD', 'BCHARTS/mrcdBRL' ]
for counter in [0, 1, 2]:
   if counter == 0:
      str_collapse = "daily"
   if counter == 1:
      str_collapse = "weekly"
   if counter == 2:
      str_collapse = "monthly"
   pnls = {i:quandl.get(i, returns="numpy", collapse=str_collapse, transformation="none") for i in symbols}
   for df_name in pnls:
      a = pnls.get(df_name)
      a = pandas.DataFrame(a)
      a['Date'] = pandas.to_datetime(a['Date'])
      a.set_index('Date', inplace=True)
#     a = a.set_index(pandas.DatetimeIndex(a['Date']))
      df_name = df_name.replace("/", "-")
      a.to_csv("{}/{}x{}.csv".format(folder, df_name, frequency[counter]), date_format='%Y-%m-%d')

