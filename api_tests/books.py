import matplotlib.pyplot
import matplotlib.animation
import time
from func_base_api import Books_exURL, plot_histogram_exBook, plotBooks

url1 = 'https://www.mercadobitcoin.net/api/BTC/orderbook/'
url2 = 'https://api.bitcointrade.com.br/v1/public/BTC/orders'

f, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, sharex = True, sharey = True)

plotBooks(url1, url2, f, ax1, ax2)
#ani = matplotlib.animation.FuncAnimation(f, plotBooks(url1, url2, f, ax1, ax2), interval = 1000, blit = True)
#matplotlib.pyplot.show()
