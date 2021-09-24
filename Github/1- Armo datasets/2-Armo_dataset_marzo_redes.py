#Genero el dataset de marzo por sobre el cual se hará la predicción

import gc
gc.collect()
import pandas as pd
import seaborn as sns
import numpy as np

#%%  marzo
marzo = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\marzo_limpiox30.csv.gz')
#Me quedo solamente con 29 dias, para estar igual con febrero
marzo = marzo.loc[marzo['day']>=3].reset_index(drop=True)
marzo['day']=marzo['day']-2

#Trabajo mejor el price
marzo = marzo.assign(current_price=marzo.groupby('currency').transform(lambda x: (x - x.min()) / (x.max()- x.min())))

 
subtest1 =  marzo[['sku', 'day', 'sold_quantity']]
subtest1= subtest1.pivot_table(index = 'sku', columns= 'day', values = 'sold_quantity').add_prefix('sales')


subtest2 =  marzo[['sku', 'day', 'current_price']]
subtest2= subtest2.pivot_table(index = 'sku', columns= 'day', values = 'current_price').add_prefix('price')

subtest3 =  marzo[['sku', 'day', 'minutes_active']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'minutes_active').add_prefix('active_time')


subtest4  = marzo[['sku', 'day', 'listing_type']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'listing_type').add_prefix('listing_type')


subtest6  = marzo[['sku', 'day', 'shipping_logistic_type']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'shipping_logistic_type').add_prefix('shipping_logistic_type')

subtest7  = marzo[['sku', 'day', 'shipping_payment']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'shipping_payment').add_prefix('shipping_payment')

final = pd.merge(subtest1, subtest2, left_index=True, right_index=True )
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)

del subtest1,subtest2,subtest3,subtest4,subtest6, subtest7


final = final.reset_index(drop = False)
final.to_csv('marzo_lgbm_price_normx30_redes.csv.gz',index=False, compression="gzip")

