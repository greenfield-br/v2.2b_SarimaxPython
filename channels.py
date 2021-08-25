from func_base_channels import trace_channels

f       = open("etc/Quandl_code.txt")
numline = len(f.readlines())
N       = 23 
index_frequency = 0

for index_symbols_Y0 in range(numline):
    bYL, bYU = trace_channels(N, index_symbols_Y0, index_frequency)
