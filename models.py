from func_base import models

order = 2 
n_b   = 1
n_c   = 5
n_k   = 1
N     = 59
horizon = 12

n_min = 1
n_max = 10
col   = 4
col_U = 5
index_symbols_U0 = 0
index_symbols_Y0 = 0
index_frequency  = 0
flag_IsPlot = 1
flag_IsEval = 0
flag_IsAll  = 0 
error_L1_forecast, error_rel_forecast, perc_dif, code_U0, code_Y0, freq, mod_structure, coefficient_list = models(flag_IsPlot, flag_IsEval, flag_IsAll, col, col_U, order, n_b, n_c, n_k, n_min, n_max, N, index_symbols_U0, index_symbols_Y0, index_frequency, horizon)
