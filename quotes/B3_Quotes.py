import bovespa

i = 'WDO'
f = 'D'
folder = './raw_data'
folder_source = 'B3'
bf = bovespa.File(folder+'/'+folder_source+'/'+'COTAHIST_A2018.TXT')
filename = folder + '/' + i + f + '_test.csv'

for rec in bf.query(stock=i):
    print('{}, {}, {}, {}, {}, {}'.format(rec.date, rec.price_open, rec.price_high, rec.price_low, rec.price_close, rec.volume))
#    b = '{}, {}, {}, {}, {}, {}'.format(rec.date, rec.price_open, rec.price_high, rec.price_low, rec.price_close, rec.volume)
#    b.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#    b.to_csv(filename)

