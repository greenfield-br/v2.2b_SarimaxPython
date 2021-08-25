import numpy as np
import matplotlib.pyplot as plot
from scipy.fftpack import fft
from func_base import list_Code, array_exQuotes, lessen, ndarray_exArix

index_symbols_U0 = 4
index_symbols_Y0 = 0
index_frequency  = 0
col   = 4
N     = 10
na    = 5
nb    = 5
nc    = 0
nk    = 1
symbols = list_Code('./etc', 'quandl_code.txt')
frequency = list_Code('./etc', 'frequency_code.txt')
code_U0 = symbols[index_symbols_U0]
code_Y0 = symbols[index_symbols_Y0]
U0c = array_exQuotes(code_U0, col, frequency[index_frequency])
Y0c = array_exQuotes(code_Y0, col, frequency[index_frequency])
U0c, Y0c, len_min = lessen(U0c, Y0c)

Y0c_vstack, hatY0c_vstack = ndarray_exArix(code_U0, code_Y0, frequency[index_frequency], col, len_min, N, na, nb, nc, nk)
hatY0c_vstack[:,:(na)] = Y0c_vstack[:,:(na)]
Y0c_vstack_fft    = np.zeros([(len_min - N), N-1])
hatY0c_vstack_fft = np.zeros([(len_min - N), N-1])

for counter in np.arange(len_min - N):
   x = Y0c_vstack[counter]
   x = x[:-1]                                   #changed
   Y0c_vstack_fft[counter] = np.abs(fft(x))
   x = hatY0c_vstack[counter]
   x = hatY0c_vstack[counter][::-1]
   x = x[1:][::-1]
   hatY0c_vstack_fft[counter] = np.abs(fft(x))
   
Y0c_vstack_fft_singleside = 2 / N * Y0c_vstack_fft[:,:N // 2]
hatY0c_vstack_fft_singleside = 2 / N * hatY0c_vstack_fft[:,:N // 2]
folder = './fft'
#color = ['#283C86', '#2B477F', '#2E5278', '#315E71', '#34696A', '#387463', '#3B805C', '#3E8B55', '#41964E', '#45A247']
color = ['#E5E5BE', '#CBD1B5', '#B2BEAD', '#98ABA5', '#7F989C', '#658594', '#4C728C', '#325F83', '#194C7B', '#003973']
Ts = 1 / 1
frequency_range = np.linspace(0, 1 / (2 * Ts), N / 2)
for counter2 in np.arange(5):
   for counter3 in [0, 1]:
      if counter3 == 0:
         array0 = Y0c_vstack_fft_singleside
         array1 = 'Y0'
         ax = plot
      if counter3 == 1:
         array0 = hatY0c_vstack_fft_singleside
         array1 = 'hatY0'
         ax = plot
      for counter in np.arange(10):
         plot_label1 = 'Sliding Window = {}-th Recent'.format(len_min - N - (counter2 * 10 + counter))
         plot.loglog(frequency_range, array0[(counter2 * 10 + counter)], label = plot_label1, color=color[counter])

      frequency_range_log_labels = [60, 48, 36, 24, 18, 12, 10, 9, 6, 5, 4, 3, 2]
      frequency_range_log = np.divide(1, frequency_range_log_labels)
      ax.xticks(frequency_range_log, frequency_range_log_labels)
      ax.xlim(1e-2,5e-1)
      label = array1
      ax.xlabel(label)
      label = code_Y0
      ax.ylabel(label)
      ax.grid(True)
      ax.legend()
      
      plot_title = 'FFT, N = {}, sampling frequency is {}'.format(N, frequency[index_frequency])
      plot.title(plot_title)
      filename = '{}/{}x{}x{}x{}_map.png'.format(folder, code_Y0, array1, N, counter2)
      plot.savefig(filename) #, bbox_inches='tight')
      plot.clf()
