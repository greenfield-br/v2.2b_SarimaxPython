import urllib.request
import json
import pandas
import numpy
import matplotlib.pyplot
import datetime
def Books_exURL(_url1, _url2):
#   print(inspect.stack()[0][3])
   url1 = _url1
   url2 = _url2
   with urllib.request.urlopen(url1) as x:
      response = urllib.request.urlopen(urllib.request.Request(url1))
      a = json.loads(x.read().decode())
      book_bids1 = a['bids']
      book_bids1 = pandas.DataFrame(book_bids1, columns = ['price', 'quantity'])
      book_asks1 = a['asks']
      book_asks1 = pandas.DataFrame(book_asks1, columns = ['price', 'quantity'])
   with urllib.request.urlopen(url2) as x:
      response = urllib.request.urlopen(urllib.request.Request(url2))
      a = json.loads(x.read().decode())
      book_bids2 = pandas.DataFrame(a['data']['bids'], columns = ['unit_price', 'amount'])
      book_asks2 = pandas.DataFrame(a['data']['asks'], columns = ['unit_price', 'amount'])
   return book_bids1, book_asks1, book_bids2, book_asks2

def array_bins_exBooks(_book_bids1, _book_asks1, _book_bids2, _book_asks2):
#   print(inspect.stack()[0][3])
   book_bids1 = _book_bids1
   book_asks1 = _book_asks1
   book_bids2 = _book_bids2
   book_asks2 = _book_asks2
   offer_margin = 0.02
   book_bids_price_max = min(book_bids1['price'].max(), book_bids2['unit_price'].max())
   book_bids_price_min = book_bids_price_max * (1 - offer_margin)
   book_asks_price_min = max(book_asks1['price'].min(), book_asks2['unit_price'].min())
   book_asks_price_max = book_asks_price_min * (1 + offer_margin)
   histogram_bins_margin = 2
   bins_bids_width = 30
   array_bins = numpy.arange(book_bids_price_min - histogram_bins_margin * bins_bids_width, book_asks_price_max + histogram_bins_margin * bins_bids_width, bins_bids_width)
   return array_bins
  

def plot_histogram_exBook(_book_bids1, _book_asks1, _book_bids2, _book_asks2, _f, _ax1, _ax2):
#   print(inspect.stack()[0][3])
   book_bids1 = _book_bids1
   book_asks1 = _book_asks1
   book_bids2 = _book_bids2
   book_asks2 = _book_asks2
   f = _f
   ax1 = _ax1
   ax2 = _ax2
   array_bins = array_bins_exBooks(book_bids1, book_asks1, book_bids2, book_asks2)

   ax1.hist(book_asks2['unit_price'], bins = array_bins, weights = book_asks2['amount'], color = '#FF0000', orientation = 'horizontal', alpha = 0.6, label = 'asks btctrade')
   ax1.hist(book_bids1['price'], bins = array_bins, weights = book_bids1['quantity'], color = '#00FF00', orientation = 'horizontal', alpha = 0.6, label = 'bids mrcd')
   ax2.hist(book_asks1['price'], bins = array_bins, weights = book_asks1['quantity'], color = '#FF0000', orientation = 'horizontal', alpha = 0.6, label = 'asks mrcd')
   ax2.hist(book_bids2['unit_price'], bins = array_bins, weights = book_bids2['amount'], color = '#00FF00', orientation = 'horizontal', alpha = 0.6, label = 'bids btctrade')

   locs = ax1.get_yticks()
   locs_max = numpy.max(locs)
   locs_min = numpy.min(locs)
   len_locs = len(locs)
   locs_step = (locs_max - locs_min) / (len_locs - 1)
   locs_step_factor = 4
   locs_step = locs_step / locs_step_factor
   locs = numpy.arange(locs_min, locs_max, locs_step)
   ax1.set_yticks(locs)

   ax1.set_xscale('log')
   timestamp = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
   title = 'Arbitrage Monitor' + ' ' + timestamp 
   ax1.set_title(title, loc = 'left')
   ax1.legend(loc = 'lower right')
   ax1.grid(b = True)
   ax2.set_xscale('log')
   ax2.legend(loc = 'lower right')
   ax2.grid(b = True)

   matplotlib.pyplot.tight_layout()
   folder   = '.'
   filename = 'book3.png'
   filename = folder + '/' + filename
   matplotlib.pyplot.savefig(filename)

def plotBooks(_url1, _url2, _f, _ax1, _ax2):
   url1 = _url1
   url2 = _url2
   f = _f
   ax1 = _ax1
   ax2 = _ax2
   book_bids1, book_asks1, book_bids2, book_asks2 = Books_exURL(url1, url2)
   plot_histogram_exBook(book_bids1, book_asks1, book_bids2, book_asks2, f, ax1, ax2)

def plot_histogram_exBook2(_book_bids1, _book_asks1, _book_bids2, _book_asks2, _f, _ax1, _ax2):
#   print(inspect.stack()[0][3])
   book_bids1 = _book_bids1
   book_asks1 = _book_asks1
   book_bids2 = _book_bids2
   book_asks2 = _book_asks2
   f = _f
   ax1 = _ax1
   ax2 = _ax2
   array_bins = array_bins_exBooks(book_bids1, book_asks1, book_bids2, book_asks2)

   ax1.hist(book_bids1['price'], bins = array_bins, weights = book_bids1['quantity'], color = '#00FF00', orientation = 'horizontal', alpha = 0.7, label = 'bids')
   ax1.hist(book_asks1['price'], bins = array_bins, weights = book_asks1['quantity'], color = '#FF0000', orientation = 'horizontal', alpha = 0.7, label = 'asks')
   ax2.hist(book_bids2['unit_price'], bins = array_bins, weights = book_bids2['amount'], color = '#00FF00', orientation = 'horizontal', alpha = 0.7, label = 'bids')
   ax2.hist(book_asks2['unit_price'], bins = array_bins, weights = book_asks2['amount'], color = '#FF0000', orientation = 'horizontal', alpha = 0.7, label = 'asks')

   locs = ax1.get_yticks()
   locs_max = numpy.max(locs)
   locs_min = numpy.min(locs)
   len_locs = len(locs)
   locs_step = (locs_max - locs_min) / (len_locs - 1)
   locs_step_factor = 3
   locs_step = locs_step / locs_step_factor
   locs = numpy.arange(locs_min, locs_max, locs_step)
   ax1.set_yticks(locs)

   ax1.set_xscale('log')
   ax1.set_title('mrcd_BTCBRL', loc = 'left')
   ax1.legend(loc = 'lower right')
   ax1.grid(b = True)
   ax2.set_xscale('log')
   ax2.set_title('btctrade_BTCBRL', loc = 'left')
   ax2.grid(b = True)

   matplotlib.pyplot.tight_layout()
   folder   = '.'
   filename = 'book3.png'
   filename = folder + '/' + filename
   matplotlib.pyplot.savefig(filename)
