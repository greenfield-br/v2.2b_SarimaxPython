import matplotlib.pyplot, numpy, inspect, datetime
from matplotlib.collections import LineCollection
converterfunc_time = lambda x : (datetime.datetime.strptime(x.decode('UTF-8'),'%Y-%m-%d'))

def array_exQuotes( Quandl_code, col, frequency):
   print(inspect.stack()[0][3])
   folder   = './raw_data'
   filename = '{}/{}x{}.csv'.format(folder, Quandl_code, frequency)
   array    = numpy.genfromtxt(filename, skip_header=1, dtype=None, converters={0:converterfunc_time}, delimiter=',')
   array    = numpy.asarray([x[col] for x in array])
   return array

def lessen(A1, A2):
   print(inspect.stack()[0][3])
   lenA1 = len(A1)
   lenA2 = len(A2)
   len_min = min(lenA1,lenA2)
   if   lenA1 > lenA2:
      A1 = A1[lenA1-len_min:]
   elif lenA1 < lenA2:
      A2 = A2[lenA2-len_min:]
   return A1, A2, len_min

def list_Code(folder, filename):
   print(inspect.stack()[0][3])
   filename = '{}/{}'.format(folder, filename)
   with open(filename, 'r') as x:
      data = x.read().split()
   return data

def crit(x, Y, N_pc):
   print(inspect.stack()[0][3])
   # diferenças da série
   Yd1  = numpy.insert( numpy.diff(Y,1), 0, numpy.nan )
   ind_pc = []
   li     = 1
   
   # repete enquanto não encontrar N_pc pontos críticos
   for aux in range(1000):
       li  = li -0.1
       lim = li*numpy.nanmax(Yd1)

       # maximos locais significativos
       indu = numpy.where( (Yd1[:-1]>lim) & (Yd1[1:]<-lim) )[0]
       Ypu  = numpy.nan*numpy.zeros(len(Y))
       Ypu[indu] = Y[indu]
       
       # minimos locais significativos
       indl = numpy.where( (Yd1[:-1]<-lim) & (Yd1[1:]>lim) )[0]
       Ypl  = numpy.nan*numpy.zeros(len(Y))
       Ypl[indl] = Y[indl]     
       ind_pc = numpy.union1d(indl, indu)
       
       if ( len(ind_pc) >= N_pc ):
           break
   
   # pontos significativos
   Y_pc   = numpy.nan*numpy.zeros(len(Y))
   Y_pc[ind_pc] = Y[ind_pc]
   #matplotlib.pyplot.plot(x, Y, 'k', linewidth=1.5)
   #matplotlib.pyplot.scatter(x, Y, c="y")
   #matplotlib.pyplot.scatter(x, Ypu, c="b")
   #matplotlib.pyplot.scatter(x, Ypl, c="r")
   #matplotlib.pyplot.xticks(rotation=30, horizontalalignment='right')
   #matplotlib.pyplot.show()
   #matplotlib.pyplot.clf()
   return Y_pc, ind_pc


def evaluate_fit(xs, Y, polynomial):
    print(inspect.stack()[0][3])
    # gera saída simulada
    Yf = polynomial(xs)
    
    # obtém pontos de inflexão
    Y_d2   = polynomial.deriv(2)
    ind_ip = numpy.rint( numpy.round(numpy.roots(Y_d2)) )
    ind_ip = ind_ip.astype(int)
    ind_ip = ind_ip.tolist()
    ind_ip = numpy.sort(ind_ip)
    ind_ip[ind_ip >= len(Y)] = len(Y)-1
    ind_ip[ind_ip <= 0] = 1
    
    # avalia qualidade do fit
    SSR = numpy.sum( (Yf -numpy.mean(Yf))**2 )
    SST = numpy.sum( (Y -numpy.mean(Y))**2 )
    r2  = SSR/SST
    return Yf, r2, ind_ip


def pivot(Ys, n):
    print(inspect.stack()[0][3])
    xs   = numpy.linspace(1,len(Ys),len(Ys))
    
    # regressão da subsérie por polinômio de 2º grau e identifica concavidade
    coefficients = numpy.polyfit(xs, Ys, 2)
    polynomial   = numpy.poly1d(coefficients)
    if ( polynomial[2] > 0 ):
        pivo  = numpy.nanmin( Ys )
    else:
        pivo  = numpy.nanmax( Ys )
    ipivo = int( numpy.where( Ys==pivo )[0][0] )
    return pivo, ipivo


def reta(ind, Y, ipivo, pivo, j):
    print(inspect.stack()[0][3])
    x      = numpy.linspace(0,len(Y)-1,len(Y))
    bcoef  = numpy.array([float('Inf'), float('Inf')])
    for i in ind:
        y        = numpy.nan*numpy.zeros(len(Y))
        y[ipivo] = pivo
        y[i]     = Y[i]
        idx  = numpy.isfinite(x) & numpy.isfinite(y)
        coef = numpy.polyfit(x[idx], y[idx], 1)
        if numpy.any( coef[j] < bcoef[j]):
            bcoef = coef   
    return bcoef


def canal_pivo_alta(ind, Y, ind_pc, ipivo, pivo):
    print(inspect.stack()[0][3])
    indip  = [i for i in ind if i <= ipivo]
    indfp  = [i for i in ind if i >= ipivo]
    indi  = [i for i in numpy.intersect1d(ind, ind_pc) if i < ipivo]
    indf  = [i for i in numpy.intersect1d(ind, ind_pc) if i > ipivo]

    coefi = reta(indi, Y, ipivo, pivo, 1)
    coeff = reta(indf, Y, ipivo, pivo, 0)
    polynomiali = numpy.poly1d(coefi)
    polynomialf = numpy.poly1d(coeff)
    Yli         = numpy.array(polynomiali( indip) )
    Ylf         = numpy.array(polynomialf( indfp) )
    Yl          = numpy.concatenate([Yli, Ylf[1:]])      
    Yui         = Yli +numpy.nanmax(Y[indip] -Yli)
    Yuf         = Ylf +numpy.nanmax(Y[indfp] -Ylf)
    Yu          = numpy.concatenate([Yui, Yuf[1:]])
    return Yl, Yu


def canal_pivo_baixa(ind, Y, ind_pc, ipivo, pivo):
    print(inspect.stack()[0][3])
    indip  = [i for i in ind if i <= ipivo]
    indfp  = [i for i in ind if i >= ipivo]
    indi  = [i for i in numpy.intersect1d(ind, ind_pc) if i < ipivo]
    indf  = [i for i in numpy.intersect1d(ind, ind_pc) if i > ipivo]
    
    coefi = reta(indi, Y, ipivo, pivo, 0)
    coeff = reta(indf, Y, ipivo, pivo, 1)
    polynomiali = numpy.poly1d( coefi )
    polynomialf = numpy.poly1d( coeff )
    Yui         = numpy.array( polynomiali(indip) )
    Yuf         = numpy.array( polynomialf(indfp) )
    Yu          = numpy.concatenate([Yui, Yuf[1:]])        
    Yli         = Yui +numpy.nanmin(Y[indip] -Yui)
    Ylf         = Yuf +numpy.nanmin(Y[indfp] -Yuf)
    Yl          = numpy.concatenate([Yli, Ylf[1:]])    
    return Yl, Yu


def perfil(Y, n):
    print(inspect.stack()[0][3])  
    xs   = numpy.linspace(1,len(Y),len(Y))
   
    # regressão da série por polinômio de grau n e identifica concavidade
    coefficients = numpy.polyfit(xs, Y, n)
    polynomial   = numpy.poly1d(coefficients)
    Yf, r2, ind_ip = evaluate_fit(xs, Y, polynomial)
    
    #matplotlib.pyplot.plot(xs, Y, 'k', xs, Yf, 'b', xs, Yf, 'b', linewidth=1.5)
    #matplotlib.pyplot.title('r squared = {}'.format(r2), loc='left')
    #matplotlib.pyplot.show()
    #matplotlib.pyplot.clf()
    return ind_ip, r2


def channels(x, Y, Y_pc, ind_pc):
    print(inspect.stack()[0][3])
    br2  = float('-Inf')
    degr = 2
    
    for aux in range(1000):
        # inicializa arrays
        YL = []
        YU = []
        
        # aproxima série completa por curva de grau n
        ind_ip, r2 = perfil(Y, degr)
        
        # reparte série em grau-1 subséries
        if numpy.any(ind_ip):
            ind_list = numpy.split( numpy.where(Y)[0], ind_ip)
        if not numpy.any(ind_ip):
            ind_list = [numpy.where(Y)[0], numpy.where(Y)[0]]

        # para cada subsérie
        for j in numpy.arange( len(ind_list) ):
            ind = ind_list[j]
            Ys  = Y[ind]
                        
            # obtém pivô da subsérie
            pivo, ipivo = pivot(Ys, 2)
            
            # aproximação por retas
            if ( pivo == numpy.nanmin(Ys) ):
                Yl, Yu = canal_pivo_alta(ind, Y_pc, ind_pc, ipivo, pivo)     
            else:
                Yl, Yu = canal_pivo_baixa(ind, Y_pc, ind_pc, ipivo, pivo)
            if (degr == 2):
                YL = Yl
                YU = Yu
            else:
                YL = numpy.append(YL, Yl)
                YU = numpy.append(YU, Yu)           
            #matplotlib.pyplot.plot(ind, Ys, 'k', linewidth=1.5)
            #matplotlib.pyplot.plot(ind, Yl, ind, Yu, color='#A2A2A2', linewidth=1.5)                  
            #matplotlib.pyplot.show()   
            #matplotlib.pyplot.clf()
            
        # verifica se melhora na qualidade do fit > 5%
        if ( r2 > 1.02*br2 ):
            bYL = YL
            bYU = YU                         
            br2 = r2
            
        #matplotlib.pyplot.plot(x, Y, 'k', linewidth=1.5)
        #matplotlib.pyplot.plot(x, YL, x, YU, color='#A2A2A2', linewidth=1.5)                  
        #matplotlib.pyplot.show()   
        #matplotlib.pyplot.clf()
        degr += 1
        if ( degr > 2 ):
            break
    return bYL, bYU

   
def trace_channels(N, index_symbols_Y0, index_frequency):   
   print(inspect.stack()[0][3])
   symbols   = list_Code('./etc', 'Quandl_code.txt')
   frequency = list_Code('./etc', 'frequency_code.txt')
   code_Y0   = symbols[index_symbols_Y0]
   code_U0   = symbols[index_symbols_Y0]
   Y0c       = array_exQuotes(code_Y0, 4, frequency[index_frequency])
   U0c       = array_exQuotes(code_U0, 5, frequency[index_frequency])
   U0, Y0, len_min = lessen(U0c, Y0c)
   x0 = array_exQuotes(code_Y0, 0, frequency[index_frequency])
   if ( N > len_min ):
       N = len_min-12
   x = x0[-N:]
   Y = Y0[-N:]
   U = U0[-N:]
   current_date = datetime.datetime.now().strftime("%m-%d-%Y %H:%M")

   # obtem pontos criticos, pivo e traça canais
   proportion_of_N = 0.8
   N_pc = round(N*proportion_of_N)
   Y_pc, ind_pc = crit(x, Y, N_pc)
   bYL, bYU = channels(x, Y, Y_pc, ind_pc)   




#   # plota série
#   fig, ax = matplotlib.pyplot.subplots( )
#   ax.plot(x, Y, color='k', linewidth=1.5)
#   small_size = 8
#   matplotlib.pyplot.title('(Y, f, N) = ({}, {}, {}) @{}'.format(code_Y0.replace("_", ""), frequency[index_frequency], N, current_date), loc='left', fontsize=small_size+2)
#   matplotlib.pyplot.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=small_size)
#   matplotlib.pyplot.setp(ax.get_yticklabels(), fontsize=small_size)
#   matplotlib.pyplot.savefig( '{}/{}x{}.png'.format('./channels', code_Y0, frequency[index_frequency]), dpi=600, bbox_inches='tight' )
#   matplotlib.pyplot.clf()
   
   # plota série com canais e distribuição
   wid     = matplotlib.pyplot.rcParams["figure.figsize"][0]
   hei     = matplotlib.pyplot.rcParams["figure.figsize"][1]
   fig, ax = matplotlib.pyplot.subplots(1, figsize=(wid, hei) )
   ax.plot(x, bYL, x, bYU, color='#A2A2A2', linewidth=1.5)
   ax.plot(x, Y, color='k', linewidth=1.5)
   hist    = numpy.histogram(Y, bins='sqrt')
   
   lines_N = []
   point_hei = hei*72
   y_low, y_high = ax.get_ylim()
   x_low, x_high = ax.get_xlim()
   for i in numpy.arange(0,len(hist[1])-1):
      y_pos   = hist[1][i] +(hist[1][i+1]-hist[1][i])/2
      lines_N.append([tuple( [x_low, x_high]), tuple([y_pos, y_pos])])
   lines_N = [list(zip(ix, iy)) for ix, iy in lines_N]
   yrange  = y_high -y_low
   linewid = hist[1][1]-hist[1][0]                # in data units
   pointlinewid = linewid*(point_hei/yrange)*0.7  # corresponding width in pts
   lines_N = LineCollection(lines_N, array=hist[0]*100/N, cmap='GnBu', linewidths=pointlinewid)
      
   ax1 = ax.add_collection(lines_N)
   ax1.set_clim(vmin=0, vmax=100)
#  matplotlib.rcParams.update({'font.size': small_size})
#  matplotlib.rcParams['font.size'] = 10
   matplotlib.pyplot.colorbar(ax1, orientation='vertical').set_label('distribution [%]')

   STD_V   = [] 
   lines_N = []
   for i in numpy.arange(0,len(U)):
      x_pos = datetime.datetime.toordinal(x[i])
      lines_N.append([tuple( [x_pos, x_pos]), tuple([y_low, y_low+(y_high-y_low)*.05] )])
      T      = len_min
      std_v  = numpy.std(U0[i:T-N+i])   # desvio padrão com T amostras
      mean_v = numpy.mean(U0[i:T-N+i])  # media com T amostras
      z_v    = ( U[i]-mean_v )/std_v    # z score volume
      STD_V  = numpy.append(STD_V, z_v)
      
   lines_N = [list(zip(ix, iy)) for ix, iy in lines_N]
   xrange  = x_high -x_low
   linewid = max( [datetime.datetime.toordinal(x[i+1]) - datetime.datetime.toordinal(x[i]) for i in numpy.arange(0,len(U)-1)] )+1
   pointlinewid = linewid*(point_hei/xrange)*0.96
   
   lines_N = LineCollection(lines_N, array=STD_V, cmap='jet', linewidths=pointlinewid)
   ax2 = ax.add_collection(lines_N)
   ax2.set_clim(vmin=0, vmax=2)
   
#  matplotlib.rcParams.update({'font.size': small_size}) 
   matplotlib.pyplot.colorbar(ax2, orientation='horizontal').set_label('relative volume in standard deviation')
   matplotlib.pyplot.xticks(rotation=30, horizontalalignment='right', fontsize='small')
   matplotlib.pyplot.yticks(fontsize='small') 
   matplotlib.pyplot.title('(Y, f, N) = ({}, {}, {}) @{}'.format(code_Y0.replace("_", ""), frequency[index_frequency], N, current_date), loc='left', fontsize='small')
   matplotlib.pyplot.savefig( '{}/{}x{}_channel.png'.format('./channels', code_Y0, frequency[index_frequency]), dpi=600, bbox_inches='tight')
   matplotlib.pyplot.clf()
   return bYL, bYU
