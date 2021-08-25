import matplotlib.pyplot, numpy, inspect, warnings, datetime, sys
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from itertools import combinations
from matplotlib.collections import LineCollection
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
converterfunc_time = lambda x : (datetime.datetime.strptime(x.decode('UTF-8'),'%Y-%m-%d'))
warnings.filterwarnings("ignore")


def array_exQuotes( Quandl_code, col, frequency):
   print(inspect.stack()[0][3])
   folder   = './raw_data'
   filename = '{}/{}x{}.csv'.format(folder, Quandl_code, frequency)
   array    = numpy.genfromtxt(filename, skip_header=1, dtype=None, converters={0:converterfunc_time}, delimiter=',')
   array    = numpy.asarray([x[col] for x in array])
   return array

def arix(order, n_b, n_c, n_k, U0, Y0):
   print(inspect.stack()[0][3])
   n_a = order
   U0, Y0, len_min = lessen(U0, Y0)
   Ud = numpy.diff(U0,1)  
   Yd = numpy.diff(Y0,1)
   
   # monta vetores de regressões
   n_max = max(n_a,n_b)
   U1 = numpy.vstack([Ud[i:i+n_max] for i in range(0,len_min-n_max-n_k)])  
   Y1 = numpy.vstack([Yd[i:i+n_max] for i in range(0,len_min-n_max-n_k)])
   if (n_k != 1): 
      Y = Yd[n_max:-n_k+1]
   else:
      Y = Yd[n_max:]
      
   # estimacao minimos quadrados
   R  = numpy.hstack((-Y1, U1))    
   RR = numpy.dot(R.T, R)           
   RY = numpy.dot(R.T, Y.T)
   coeff = numpy.dot(numpy.linalg.inv(RR), RY)
   
   # monta vetores de parametros
   coeff_a = numpy.flipud(-coeff[0:n_a].T)    
   coeff_b = numpy.flipud( coeff[n_a:n_a+n_b].T)
   if (n_b > n_a):
       coeff_a = numpy.pad(coeff_a, (0,n_b-n_a), 'constant') 
   else:
       coeff_b = numpy.pad(coeff_b, (0,n_a-n_b), 'constant')
     
   # prediz novas diferenças da saida    
   conv_y  = numpy.convolve( Yd, coeff_a, 'valid' )
   conv_u  = numpy.convolve( Ud, coeff_b, 'valid' )
   hatYd   = conv_y +conv_u
   
   # converte em prediçoes da saida
   hatY0   = numpy.concatenate( [Y0[:n_max+1], hatYd +Y0[n_max:]] )
   Y0      = numpy.append(Y0, numpy.nan)
   return coeff_a, coeff_b, hatY0, Y0

def arimax(order, n_b, n_c, n_k, U0, Y0):
   print(inspect.stack()[0][3])
   n_a = order
   U0, Y0, len_min = lessen(U0, Y0)

   # estima modelo 
   U = numpy.vstack([numpy.roll(U0, i) for i in range(0,n_b)])
   for i in range(0,n_b):
       U[i,:i] = 0
   U0 = U.T
   model = ARIMA(Y0, (n_a,1,n_c), U0)
   model_fit = model.fit(transparams=True, disp=0, method='css')
   coeff_a = ARIMAResults.arparams
   coeff_b = ARIMAResults.maparams
   coeff_c = ARIMAResults.maparams

   # prediçoes+forecast da saida
   n_max = max(n_a,n_b)
   hatY0 = model_fit.predict(typ='levels', end=len_min, exog=U0[-1,:])
   hatY0 = numpy.concatenate( [Y0[:n_max+1], hatY0] )
   Y0    = numpy.append(Y0, numpy.nan)
   return coeff_a, coeff_b, coeff_c, hatY0, Y0

def sarimax(order, n_b, n_c, n_k, U0, Y0):
   print(inspect.stack()[0][3])
   n_a = int(order)
   n_b = int(n_b)
   n_c = int(n_c)
   U0, Y0, len_min = lessen(U0, Y0)
   
   # estima modelo
   Ue = numpy.vstack([numpy.roll(U0, i) for i in range(0,n_b)])
   for i in range(0,n_b):
       Ue[i,:i] = 0
   U0 = Ue.T
   model = SARIMAX(Y0, U0, (n_a,1,n_c), enforce_stationarity=False, enforce_invertibility=False)
   model_fit = model.fit(disp=0) # model.fit()
   coeff_a = 1
   coeff_b = 1
   coeff_c = 1
   
   # prediçoes+forecast da saida
   hatY0 = model_fit.predict(typ='levels', end=len_min, exog=numpy.array([U0[-1,:]]))
   Y0    = numpy.append(Y0, numpy.nan)
   return coeff_a, coeff_b, coeff_c, hatY0, Y0

def arix_exCode(U0, Y0, order, n_b, n_c, n_k, N, counter):
   print(inspect.stack()[0][3])
   U0_sliding_window = U0[counter:(N + counter)]
   Y0_sliding_window = Y0[counter:(N + counter)]  
   if (n_c == 0):
       coeff_a, coeff_b, hatY0_sliding_window, Y0_sliding_window = arix(order, n_b, n_c, n_k, U0_sliding_window, Y0_sliding_window)
       mod_structure = 'arix'
   else:
       #coeff_a, coeff_b, coeff_c, hatY0_sliding_window, Y0_sliding_window = arimax(order, n_b, n_c, n_k, U0_sliding_window, Y0_sliding_window)
       #mod = 'arima'
       coeff_a, coeff_b, coeff_c, hatY0_sliding_window, Y0_sliding_window = sarimax(order, n_b, n_c, n_k, U0_sliding_window, Y0_sliding_window)
       mod_structure = 'sarimax'
   return coeff_a, coeff_b, hatY0_sliding_window, Y0_sliding_window, mod_structure

def codeUY(code_U0, code_Y0):
   if code_U0 == code_Y0:
       code = code_Y0
   else:
       code = code_U0 +'x' +code_Y0
   return code

def corr(code_U0, code_Y0, col, col_U, frequency):
   print(inspect.stack()[0][3])
   U0 = array_exQuotes(code_U0, col_U, frequency)
   Y0 = array_exQuotes(code_Y0, col, frequency)
   U0, Y0, len_min = lessen(U0,Y0)
   mtx_corr = numpy.zeros( (len_min-1, len_min-1) ) 
   for i in range(0,len_min-1):                  
      for j in range(0,len_min-1-i):             
         mtx_corr[i,i+j] = numpy.corrcoef(U0[j:j+i+2],Y0[j:j+i+2])[1,0]
   return mtx_corr

def not_complicated_corr(code_U0, code_Y0, col, col_U, frequency, N, horizon):
   print(inspect.stack()[0][3])
   Us = array_exQuotes(code_U0, col_U, frequency)[-(N+horizon):]
   Ys = array_exQuotes(code_Y0, col, frequency)[-(N+horizon):]
   corr_N = numpy.zeros( horizon+1 ) 
   for i in range(0,horizon+1):
       corr_N[i] = numpy.corrcoef(Us[i:N+i],Ys[i:N+i])[1,0]
   return corr_N

def corr_plot(col, col_U, index_frequency, folder, symbols, frequency):#, N):
   print(inspect.stack()[0][3])
   symbols   = tuple(combinations(symbols, 2))
   for i in numpy.arange( len(symbols)-1 ):
      mtx_corr = corr(symbols[i][0], symbols[i][1], col, col_U, frequency[index_frequency]) #, N)
      matplotlib.pyplot.imshow(mtx_corr, cmap='jet', origin='lower')
      matplotlib.pyplot.ylabel(symbols[i][1])
      matplotlib.pyplot.xlabel(symbols[i][0])
      matplotlib.pyplot.clim(-1,1)
      matplotlib.pyplot.colorbar(orientation='vertical').set_label('Correlation Coefficient')
      filename = '{}/{}x{}_map.png'.format(folder, symbols[i][1], symbols[i][0])
      matplotlib.pyplot.savefig(filename, dpi=600, bbox_inches='tight')
      matplotlib.pyplot.clf()

def error_exForecast(N_min, N_max, n_max_combination, horizon, len_min, n_k, N, U0c, Y0c, Ymx, Ydx):
   print(inspect.stack()[0][3])
   coefficient_list   = numpy.empty((0,3), int)
   error_L0_forecast  = numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   error_L1_forecast  = numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   error_rel_forecast = numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   hatY0N             = numpy.nan*numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   hatYmx             = numpy.nan*numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   hatYdx             = numpy.nan*numpy.zeros(shape=((n_max_combination),(horizon+1)), dtype=float)
   Y0c = numpy.append(Y0c, numpy.nan)   
   Y0d = numpy.insert( numpy.diff(Y0c,1), 0, numpy.zeros(1) ) 
   
   for counter_a in numpy.arange(N_min[0], N_max[0]+1):
       for counter_b in numpy.arange(N_min[1], N_max[1]+1):
           for counter_c in numpy.arange(N_min[2], N_max[2]+1):
               for counter1 in numpy.arange((len_min -N -horizon), (len_min -N +1)):
                   counter2 = coefficient_list.shape[0]    
                   coeff_a, coeff_b, hatY0c_sliding_window, Y0c_sliding_window, mod_structure = arix_exCode(U0c, Y0c, counter_a, counter_b, counter_c, n_k, N, counter1)
#                   coeff_a, coeff_b, hatYmx_sliding_window, Ymx_sliding_window, mod_structure = arix_exCode(U0c, Ymx, counter_a, counter_b, counter_c, n_k, N, counter1)
#                   coeff_a, coeff_b, hatYdx_sliding_window, Ydx_sliding_window, mod_structure = arix_exCode(Y0d, Ydx, counter_a, counter_b, counter_c, n_k, N, counter1)                  
                   hatY0N[(counter2), (counter1 - (len_min -N -horizon))] = hatY0c_sliding_window[-1]
                   hatYmx[(counter2), (counter1 - (len_min -N -horizon))] = hatY0c_sliding_window[-1] #hatYmx_sliding_window[-1]
                   hatYdx[(counter2), (counter1 - (len_min -N -horizon))] = hatY0c_sliding_window[-1] #hatYdx_sliding_window[-1]
                   error_L1_forecast[(counter2), (counter1 -(len_min - N - horizon))] = numpy.absolute(Y0c[counter1+N] -hatY0c_sliding_window[-1])                      #absolute error
                   error_L0_forecast[(counter2), (counter1 -(len_min - N - horizon))] = Y0c[counter1+N] -hatY0c_sliding_window[-1]                                      #error
                   error_rel_forecast[(counter2), (counter1 -(len_min - N - horizon))] = numpy.absolute(Y0c[counter1+N] -hatY0c_sliding_window[-1])/Y0c[counter1+N]     #relative error      
               data = numpy.asarray([counter_a, counter_b, counter_c])
               coefficient_list = numpy.vstack((coefficient_list, data))
               
   error_L0_forecast  = error_L0_forecast[:,:-1]
   error_L1_forecast  = error_L1_forecast[:,:-1]
   error_rel_forecast = error_rel_forecast[:,:-1]
   hatY0N[:,0] = numpy.nan
   hatYmx[:,0] = numpy.nan
   hatYdx[:,0] = numpy.nan
   return error_L1_forecast, error_L0_forecast, error_rel_forecast, hatY0N, hatYmx, hatYdx, mod_structure, coefficient_list

def x_axis_delta(code_Y0, frequency):
   x = array_exQuotes(code_Y0, 0, frequency)
   if (frequency == 'D'):
       x   = numpy.append(x, x[-1]+relativedelta(days=+1))
   elif (frequency == 'W'):
       if (code_Y0 == 'BTCUSD_') or (code_Y0 == 'ETHUSD_') or (code_Y0 == 'XRPUSD_'):
          x   = numpy.append(x, x[-1]+relativedelta(weeks=+1))
       else:
#        x   = [x[i] +relativedelta(weeks=+1, days=-1) for i in range(0,len(x))]
        x   = numpy.append(x, x[-1]+relativedelta(weeks=+1))
   elif (frequency == 'M'):
#       x   = [x[i] +relativedelta(months=+1, days=-1) for i in range(0,len(x))]
       x   = numpy.append(x, x[-1] +relativedelta(months=+1))
   elif (frequency == 'A'):
#       x   = [x[i] +relativedelta(years=+1, days=-1) for i in range(0,len(x))]
       x   = numpy.append(x, x[-1] +relativedelta(years=+1))
   return x
    
    
def forecast_plot(code_U0, code_Y0, frequency, len_min, N, horizon, model, Y0c_last_window, error_L1, error_Linf, error_L0_forecast, error_rel, perc_dif, hatY0N, hatYmx, hatYdx, col, col_U, Y0c, Y0h, Y0l, Ymx, mod_structure):
   print(inspect.stack()[0][3])
   error_L1 = numpy.ones([horizon+1,1])*error_L1
   x = x_axis_delta(code_Y0, frequency)[-N-1:]
   forecast_date = x[-1].strftime("%m-%d-%Y")
   evr           = 100*error_rel/numpy.mean(perc_dif)
   
   # forecast plot
   wid     = matplotlib.pyplot.rcParams["figure.figsize"][0]
   hei     = matplotlib.pyplot.rcParams["figure.figsize"][1]
   fig, ax = matplotlib.pyplot.subplots( 1, figsize=(wid, hei) )
   small_size = 8
   ax.plot(x[-(horizon+1):], Y0c_last_window[-(horizon+1):], color='k', linewidth=1.5, label='close') 
   ax.plot(x[-(horizon+1):], hatY0N, linewidth=2, color='#A2A2A2')     
   ax.errorbar(x[-(horizon+1):], numpy.concatenate([numpy.nan*numpy.zeros(1), hatY0N[1:]]), yerr=error_L1, capsize=3, errorevery=horizon, markeredgewidth=2, linewidth=2, color='#A2A2A2', label='forecast')           
                 
   # moving correlation coefficient
   corr_N = not_complicated_corr(code_U0, code_Y0, col, col_U, frequency, N, horizon)
   corr_N = corr_N[-(horizon+1):]
   corr_N = numpy.append(corr_N, 0)
   
   # vertical bars
   point_hei = hei*72
   y_low, y_high = ax.get_ylim()
   x_low, x_high = ax.get_xlim()
   lines_N = []
   for i in numpy.arange(N-horizon,N+1):
      x_pos = datetime.datetime.toordinal(x[i])
      lines_N.append([tuple([x_pos, x_pos]), tuple([y_low, y_high])])
   lines_N = [list(zip(x, y)) for x, y in lines_N]
   xrange  = x_high -x_low
   linewid = max([datetime.datetime.toordinal(x[i+1]) - datetime.datetime.toordinal(x[i]) for i in numpy.arange(N-horizon,N-1)])
   
   # forecast figure
   pointlinewid = linewid*(point_hei/xrange)*0.96
   lines_N = LineCollection(lines_N, array=corr_N, cmap='jet', linewidths=pointlinewid)
   ax1 = ax.add_collection(lines_N)
   ax1.set_clim(vmin=-1, vmax=1)
   matplotlib.rcParams.update({'font.size': small_size}) 
   label_bar = 'Correlation Coefficient'
   matplotlib.pyplot.colorbar(ax1, orientation='vertical').set_label(label_bar)
   matplotlib.pyplot.xticks(rotation=30, horizontalalignment='right', fontsize=small_size)
   matplotlib.pyplot.yticks(fontsize=small_size)
   matplotlib.pyplot.grid(True)
   matplotlib.pyplot.legend()
   ax.errorbar(x[-(horizon+1):], numpy.concatenate([numpy.nan*numpy.zeros(1), hatY0N[1:]]), yerr=error_L1, capsize=3, errorevery=horizon, markeredgewidth=2, linewidth=2, color='#A2A2A2')           
   
   if ( hatY0N[-1] > 999 ):
       precision=1
   elif ( hatY0N[-1] > 99 ):
       precision=2
   elif ( hatY0N[-1] > 9 ):
       precision=3
   else:
       precision=4
   forecast_value = hatY0N[-1]
   forecast_value = Decimal(forecast_value).quantize(Decimal(10)**-precision)
   bottom_range = hatY0N[-1] -error_L1[0]
   bottom_range = numpy.asscalar(bottom_range)
   bottom_range = round(bottom_range,precision)
   bottom_range = Decimal(bottom_range).quantize(Decimal(10)**-precision)
   top_range = hatY0N[-1] + error_L1[0]
   top_range = numpy.asscalar(top_range)
   top_range = round(top_range,precision)
   top_range = Decimal(top_range).quantize(Decimal(10)**-precision)
   matplotlib.pyplot.ylabel(code_Y0.replace("_", ""))   
   current_date = datetime.datetime.now().strftime("%m-%d-%Y %H:%M")
   model     = str(model).replace('[','').replace(' ','_').replace(']','')
   plot_title = '(X, f, N, order) = ({}, {}, {}, {}) @{} \n{}@{} in [{}, {}], (MAE, EVR) = ({:.2f}%, {:.1f}%)'.format(code_U0.replace("_", ""), frequency, N, model.split('_')[0], current_date, forecast_value, forecast_date, bottom_range, top_range, error_rel, evr, fontsize=small_size+2)
   matplotlib.pyplot.title(plot_title, loc='left')
   raw_data_folder = './forecasts'
   code = codeUY(code_U0, code_Y0)
   filename = '{}/forecast_{}x{}x{}_{}.png'.format(raw_data_folder, code, frequency, model, mod_structure)
   matplotlib.pyplot.savefig(filename, dpi=600, bbox_inches='tight')   
   matplotlib.pyplot.clf()

   # forecast figure clean
   fig, ax = matplotlib.pyplot.subplots()
   csfont  = {'fontname':'Pacifico'}
   ax.plot(x[-(horizon+1):], Y0c_last_window[-(horizon+1):], color='k', linewidth=1.5, label='close') 
   ax.plot(x[-(horizon+1):], hatY0N, linewidth=2, color='g')     
   ax.errorbar(x[-(horizon+1):], numpy.concatenate([numpy.nan*numpy.zeros(1), hatY0N[1:]]), yerr=error_L1, capsize=3, errorevery=horizon, markeredgewidth=2, linewidth=2, color='g', label='forecast')           
   yTicks = ax.get_yticks()
   xTicks = ax.get_xticks()[:-1]   
   ax.bar(xTicks, height=[max(yTicks)-min(yTicks)]*len(xTicks), width=(xTicks[1]-xTicks[0]), bottom=[min(yTicks)]*len(xTicks), align='edge', color=['#DBDDC8','#C7C9B4'])
   plot_title = '{}'.format(code_Y0.replace("_", ""), fontsize=small_size+2)
   matplotlib.pyplot.title(plot_title, **csfont, loc='right')
   matplotlib.pyplot.xticks(rotation=30, horizontalalignment='right', fontsize=small_size)
   raw_data_folder = './forecasts'
   filename = '{}/clean_forecast_{}x{}x{}_{}.png'.format(raw_data_folder, code, frequency, model, mod_structure)
   ax.yaxis.grid(True, color='w')
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['bottom'].set_visible(False)
   ax.spines['left'].set_visible(False)
   matplotlib.pyplot.savefig(filename, facecolor='#DBDDC8', transparent=True, dpi=600, bbox_inches='tight')   
   matplotlib.pyplot.clf()

   # range forecast figure   
   Y0h      = numpy.append(Y0h, numpy.nan)
   Y0l      = numpy.append(Y0l, numpy.nan)
   hatY0l   = hatYmx -hatYdx
   hatY0h   = hatYmx +hatYdx
   fig, ax = matplotlib.pyplot.subplots()
   ax.errorbar(x,  Y0c_last_window[-N-1:], fmt='none', yerr=[Y0c_last_window[-N-1:]-Y0l[-N-1:], Y0h[-N-1:]-Y0c_last_window[-N-1:]], ecolor='g', elinewidth=1.25, label='range')
   ax.errorbar(x[-(horizon+1):],  hatY0N, fmt='none', yerr=[hatY0N-hatY0l, hatY0h-hatY0N], ecolor='k', elinewidth=0.5, label='forecast')
   plot_title = '(X, f, N, order) = ({}, {}, {}, {}) @{}'.format(code_U0.replace("_", ""), frequency, N, model.split('_')[0], current_date, fontsize=small_size+2)
   matplotlib.pyplot.title(plot_title, loc='left')
   matplotlib.pyplot.xticks(rotation=30, horizontalalignment='right', fontsize=small_size)
   matplotlib.pyplot.yticks(fontsize=small_size)
   filename = '{}/range_{}x{}x{}.png'.format(raw_data_folder, code, frequency, model)
   matplotlib.pyplot.grid(True)
   matplotlib.pyplot.legend()
   matplotlib.pyplot.savefig(filename, dpi=600, bbox_inches='tight')
   matplotlib.pyplot.clf()

def forecastEval_plot(error_rel_forecast, len_min, N, code_U0, code_Y0, frequency, mod_structure, coefficient_list, model):
   print(inspect.stack()[0][3])
   if coefficient_list.shape[0] > 1:
       coefficient_list_reduced   = coefficient_list[(coefficient_list[:,0] == coefficient_list[:,1]) & (coefficient_list[:,0] == coefficient_list[:,2])]
       ind_reduced                = numpy.where( (coefficient_list[:,0] == coefficient_list[:,1]) & (coefficient_list[:,0] == coefficient_list[:,2]) )[0]
       error_rel_forecast_reduced = error_rel_forecast[ind_reduced,:]
   else:
       coefficient_list_reduced   = coefficient_list
       error_rel_forecast_reduced = error_rel_forecast
   
   fig, ax = matplotlib.pyplot.subplots(1,1)
   img     = ax.imshow(error_rel_forecast_reduced*100, cmap='jet', origin='lower')
   ax.set_xticks(numpy.arange( error_rel_forecast_reduced.shape[1] ))
   ax.set_yticks(numpy.arange( coefficient_list_reduced.shape[0] ))
   ax.set_xticklabels(range(1,error_rel_forecast_reduced.shape[1]+1))
   ax.set_yticklabels(coefficient_list_reduced)
   label_bar = 'in % of i-th Month Close'
   matplotlib.pyplot.colorbar(img, ax=ax, orientation='vertical').set_label(label_bar)
   matplotlib.pyplot.ylabel('Model Order')
   plot_xlabel = 'Sliding Window Shift. Sampling is {}'.format(frequency)
   matplotlib.pyplot.xlabel(plot_xlabel)
   current_date = datetime.datetime.now().strftime("%m-%d-%Y %H:%M")
   plot_title = '1-Step Ahead Error \n(X, Y, N) = ({}, {}, {}) @{}'.format(code_U0.replace("_", ""), code_Y0.replace("_", ""), N, current_date)
   matplotlib.pyplot.title(plot_title, loc='left')
   raw_data_folder = './forecasts'
   code = codeUY(code_U0, code_Y0)
   model    = str(model).replace('[','').replace(' ','_').replace(']','')
   filename = '{}/emap_{}x{}x{}_{}.png'.format(raw_data_folder, code, frequency, model, mod_structure)
   matplotlib.pyplot.savefig(filename, dpi=600, bbox_inches='tight')
   matplotlib.pyplot.clf()
   
   if ( numpy.max(error_rel_forecast_reduced) > 0.05 ):
       error_rel_forecast_reduced[error_rel_forecast_reduced > 0.05] = 0.05 
       fig, ax = matplotlib.pyplot.subplots(1,1)    
       img     = ax.imshow(error_rel_forecast_reduced*100, cmap='jet', origin='lower')
       ax.set_xticks(numpy.arange( error_rel_forecast_reduced.shape[1] ))
       ax.set_yticks(numpy.arange( coefficient_list_reduced.shape[0] ))
       ax.set_xticklabels(range(1,error_rel_forecast_reduced.shape[1]+1))
       ax.set_yticklabels(coefficient_list_reduced)
       matplotlib.pyplot.colorbar(img, ax=ax, orientation='vertical').set_label(label_bar)
#      ax.axes().set_aspect('auto')
       matplotlib.pyplot.ylabel('Model Order')
       matplotlib.pyplot.xlabel(plot_xlabel)
       matplotlib.pyplot.title(plot_title, loc='left')
       filename = '{}/emap_{}x{}x{}_{}_cap.png'.format(raw_data_folder, code, frequency, model, mod_structure)
       matplotlib.pyplot.savefig(filename, dpi=600, bbox_inches='tight')
       matplotlib.pyplot.clf()   

def list_Code(folder, filename):
   print(inspect.stack()[0][3])
   filename = '{}/{}'.format(folder, filename)
   with open(filename, 'r') as x:
      data = x.read().split()
   return data

def lessen(A1, A2):
   print(inspect.stack()[0][3])
   len_min = min( len(A1), len(A2) )
   A1 = A1[-len_min:]
   A2 = A2[-len_min:]
   return A1, A2, len_min

def resize_N_horizon(N, horizon, len_min):
   if N+horizon >= len_min:
      horizon = 12
      N = len_min -horizon
   return N, horizon

def models(flag_IsPlot, flag_IsEval, flag_IsAll, col, col_U, order, n_b, n_c, n_k, n_min, n_max, N, index_symbols_U0, index_symbols_Y0, index_frequency, horizon):
   print(inspect.stack()[0][3])
   symbols   = list_Code('./etc', 'Quandl_code.txt')
   frequency = list_Code('./etc', 'frequency_code.txt')
   code_U0   = symbols[index_symbols_U0]
   code_Y0   = symbols[index_symbols_Y0]
   U0c       = array_exQuotes(code_U0, col_U, frequency[index_frequency])
   Y0c       = array_exQuotes(code_Y0, col, frequency[index_frequency])
   U0c, Y0c, len_min = lessen(U0c, Y0c)
   N, horizon = resize_N_horizon(N, horizon, len_min)
   Y0h       = array_exQuotes(code_Y0, 2, frequency[index_frequency])[-len_min:]
   Y0l       = array_exQuotes(code_Y0, 3, frequency[index_frequency])[-len_min:]
   Ymx       = (Y0h +Y0l)/2
   Ydx       = (Y0h -Y0l)/2
   
   if flag_IsAll == 1:
       N_min = [n_min, n_min, n_min]
       N_max = [n_max, n_max, n_max]
       n_max_combination  = (max(N_max) -min(N_min) +1)**3
   else:
       N_min = [order, n_b, n_c]
       N_max = [order, n_b, n_c]  
       n_max_combination  = 1
   error_L1_forecast, error_L0_forecast, error_rel_forecast, hatY0N, hatYmx, hatYdx, mod_structure, coefficient_list = error_exForecast(N_min, N_max, n_max_combination, horizon, len_min, n_k, N, U0c, Y0c, Ymx, Ydx)
   try:
       index_chosen = numpy.where(( coefficient_list == [order, n_b, n_c]).all(axis=1) )[0].item()
   except ValueError:
       print("You must choose coefficients inside the n_min to n_max range")
       sys.exit(0)
       
   Y0c_last_window = numpy.append(Y0c[-len_min:], numpy.nan)
   hatY0N     = hatY0N[index_chosen,:]
   hatYmx     = hatYmx[index_chosen,:]
   hatYdx     = hatYdx[index_chosen,:]
   error_L1   = numpy.sum(error_L1_forecast[index_chosen,1:])/len(error_L1_forecast[index_chosen,1:])                 #MAE
   error_Linf = numpy.max(error_L0_forecast[index_chosen,:])                                                          #MaxAE
   error_rel  = 100*numpy.nansum(error_rel_forecast[index_chosen -n_min,1:])/len(error_rel_forecast[index_chosen,1:]) #MAPE
   perc_dif   = 100*numpy.abs( Y0c_last_window[1:-1]-Y0c_last_window[:-2] )/Y0c_last_window[:-2]                      #MAPD
   if flag_IsPlot == 1:
      forecast_plot(code_U0, code_Y0, frequency[index_frequency], len_min, N, horizon, coefficient_list[index_chosen], Y0c_last_window, error_L1, error_Linf, error_L0_forecast, error_rel, perc_dif, hatY0N, hatYmx, hatYdx, col, col_U, Y0c, Y0h, Y0l, Ymx, mod_structure)
   if flag_IsEval == 1:
      forecastEval_plot(error_rel_forecast, len_min, N, code_U0, code_Y0, frequency[index_frequency], mod_structure, coefficient_list, coefficient_list[index_chosen])
   return error_L1_forecast, error_rel_forecast, perc_dif, code_U0, code_Y0, frequency[index_frequency], mod_structure, coefficient_list


def ndarray_exArix(code_U0, code_Y0, frequency, col, len_min, N, order, n_b, n_c, n_k):
   print(inspect.stack()[0][3])
   Y0c_vstack = numpy.zeros(shape=((len_min - N),(N)), dtype=float)
   hatY0c_vstack = numpy.zeros(shape=((len_min - N),(N)), dtype=float)
   for counter in numpy.arange(len_min - N):
      coeff_a, coeff_b, hatY0c_vstack[counter], Y0c_vstack[counter], mod_structure = arix_exCode(code_U0, code_Y0, col, frequency, order, n_b, n_c, n_k, N, counter)
   return Y0c_vstack, hatY0c_vstack
