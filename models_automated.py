import numpy
from func_base import models

def table_emap(code_Y0, freq, n_min, n_combination, error_rel_forecast, error_rel, coefficient_list):
    fname = 'results/error_map_' +code_Y0 +'x' +freq +'x' +str(N) +'.csv'
    label = numpy.column_stack(['Forex', 'order', 'n_b', 'n_c', [numpy.arange(1,error_rel_forecast.shape[1]+1)], 'MAE'])
    data1 = numpy.column_stack([ numpy.vstack(numpy.repeat(code_Y0, n_combination-n_min+1)), coefficient_list.astype(int), 100*error_rel_forecast, numpy.vstack(error_rel)])
    data  = numpy.row_stack([label, data1])
    numpy.savetxt(fname, data, delimiter=",", fmt='%s')
    
def table_asset(code_Y0, freq, n_min, n_combination, error_rel, evr, coefficient_list, N):
    fname = 'results/' +code_Y0 +'x' +freq +'x' +str(N) +'.csv'
    label = numpy.column_stack( ['Forex', 'MAE', 'EVR', 'order', 'n_b', 'n_c', 'frequency', 'N'] )
    data1 = numpy.column_stack( [ numpy.vstack(numpy.repeat(code_Y0, n_combination-n_min+1)), numpy.vstack(error_rel), numpy.vstack(evr), coefficient_list, numpy.vstack(numpy.repeat(freq, n_combination-n_min+1)), numpy.vstack(numpy.repeat(N, n_combination-n_min+1)) ])
    data  = numpy.row_stack([label, data1])
    numpy.savetxt(fname, data, delimiter=",", fmt='%s')
    
def table_best(code_Y0, freq, n_min, n_combination, error_rel, evr, index_best, coefficient_list, N):
    fname = 'results/best_' +code_Y0 +'x' +freq +'x' +str(N) +'.csv'
    label = numpy.column_stack( ['Forex', 'MAE', 'EVR', 'order', 'n_b', 'n_c', 'frequency', 'N'] )
    data1 = numpy.column_stack([ code_Y0, error_rel[index_best], evr[index_best], coefficient_list[index_best,:], freq, N] )
    data  = numpy.row_stack([label, data1])
    numpy.savetxt(fname, data, delimiter=",", fmt='%s')


file_code    = open("etc/Quandl_code.txt")
numline = len(file_code.readlines())
Ni      = [59] #[59, 89, 119, 179]
Fi      = [0] #[0, 1, 2]
horizon = 12

n_min   = 1
n_max   = 6
col     = 4
col_U   = 5
flag_IsPlot = 0
flag_IsAll  = 1

for index_frequency in Fi:
    for N in Ni:                           # repeat for all N windows
        for i in range(numline):           # repeat for all assets in .txt list
            index_symbols_U0 = i
            index_symbols_Y0 = i

            for j in range(1,n_max+1):     # repeat from order 1 to n_max
                order = j
                n_b   = j
                n_c   = j
                n_k   = 1

                
                if (j == n_max ):          # verify if last iteration then plot error map
                    n_min = 1
                    flag_IsEval = 1
                    error_L1_forecast, error_rel_forecast, perc_dif, code_U0, code_Y0, freq, mod_structure, coefficient_list = models(flag_IsPlot, flag_IsEval, flag_IsAll, col, col_U, order, n_b, n_c, n_k, n_min, n_max, N, index_symbols_U0, index_symbols_Y0, index_frequency, horizon)                    
                    error_rel     = 100*numpy.sum(error_rel_forecast, axis=1)/error_rel_forecast.shape[1]
                    index_best    = numpy.where(error_rel == error_rel.min())[0]
                    evr           = 100*error_rel/numpy.mean(perc_dif)
                    table_emap(code_Y0, freq, n_min, coefficient_list.shape[0], error_rel_forecast, error_rel, coefficient_list)
                    table_asset(code_Y0, freq, n_min, coefficient_list.shape[0], error_rel, evr, coefficient_list, N)
                    table_best(code_Y0, freq, n_min, coefficient_list.shape[0], error_rel, evr, index_best, coefficient_list, N)
                else:
                    flag_IsEval = 0
                    error_L1_forecast, error_rel_forecast, perc_dif, code_U0, code_Y0, freq, mod_structure, coefficient_list = models(flag_IsPlot, flag_IsEval, flag_IsAll, col, col_U, order, n_b, n_c, n_k, n_min, n_max, N, index_symbols_U0, index_symbols_Y0, index_frequency, horizon)
