import numpy
from func_base import list_Code, corr_plot#, corr_N_plot

flag_corr_N     = 0
index_frequency = 0
col   = 4
col_U = 4
N   = 100

symbols = list_Code('./etc', 'Quandl_code.txt')
frequency = list_Code('./etc', 'frequency_code.txt')
if flag_corr_N == 0:
   folder = './maps'
   corr_plot(col, col_U, index_frequency, folder, symbols, frequency)#, N)
if flag_corr_N == 1:
   for counter in numpy.arange(len(symbols)):
      folder = './maps_N'
#      corr_N_plot(counter, col, col_U, index_frequency, N, folder, frequency)
