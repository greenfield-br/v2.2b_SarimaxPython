import numpy, datetime, os, re
str2date = lambda x : datetime.datetime.strptime(x.decode('UTF-8'),'%d.%m.%Y').strftime('%Y-%m-%d')

def table_read(l1, l2, f):
    if (l2 != ''):
        filename = './results/' +l1 +'_' +l2 +'x' +f
        #label   = numpy.column_stack( ['Forex', 'MAPE', 'EVR', 'order', 'frequency'] )
        data     = numpy.genfromtxt(filename, skip_header=1, dtype=None, delimiter=',', usecols=(0,1,2,3,4))
        return data
        
def table_best(l1, l2, f, data):
        fname = 'results/best_' +l1 +'_' +l2 +'x' +f +'.csv'
        numpy.savetxt(fname, data, delimiter=",", fmt='%s')

DATA = []
path = './results/'
for file in os.listdir(path):
    if file.endswith(".csv") and os.path.isfile(os.path.join(path,file)) and 'best' in file:
        sfile = re.split('-|x|_|.',file)
        l1 = sfile[1]
        l2 = sfile[2]
        f  = sfile[3]
        print(l1, l2, f)
        data = table_read(l1, l2,f)
        DATA = numpy.row_stack([DATA, data])
        
    if file.endswith(".csv") and os.path.isfile(os.path.join(path,file)) and 'best' in file:
       
