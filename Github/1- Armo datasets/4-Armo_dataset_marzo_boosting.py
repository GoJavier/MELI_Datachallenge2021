#Creo el dataset para la predicción del boosting

import gc
gc.collect()
import pandas as pd
import seaborn as sns
import numpy as np

#%%  marzo
marzo = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\marzo_limpio.csv.gz')
marzo = marzo.loc[marzo['day']>=4].reset_index(drop=True)
marzo['day']=marzo['day']-3

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

#%% Promedios cada 3 dias
marzo_test = marzo.sort_values(['sku','day']).reset_index(drop=True).copy()

marzo_test['promedio_3'] = marzo.groupby(['sku'])['sold_quantity'].rolling(3, min_periods=3).mean().reset_index(drop=True)
marzo_test['promedio_7'] = marzo.groupby(['sku'])['sold_quantity'].rolling(7, min_periods=7).mean().reset_index(drop=True)
marzo_test['promedio_15'] = marzo.groupby(['sku'])['sold_quantity'].rolling(15, min_periods=15).mean().reset_index(drop=True)
marzo_test['promedio_20'] = marzo.groupby(['sku'])['sold_quantity'].rolling(20, min_periods=20).mean().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  marzo_test[['sku', 'day', 'promedio_3']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'promedio_3', dropna=False).add_prefix('promedio_3')
subtest4  = marzo_test[['sku', 'day', 'promedio_7']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'promedio_7', dropna=False).add_prefix('promedio_7')
subtest6  = marzo_test[['sku', 'day', 'promedio_15']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'promedio_15', dropna=False).add_prefix('promedio_15')
subtest7  = marzo_test[['sku', 'day', 'promedio_20']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'promedio_20', dropna=False).add_prefix('promedio_20')

final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7

marzo_test['promedio_3_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(3, min_periods=3).mean().reset_index(drop=True)
marzo_test['promedio_7_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(7, min_periods=7).mean().reset_index(drop=True)
marzo_test['promedio_15_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(15, min_periods=15).mean().reset_index(drop=True)
marzo_test['promedio_20_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(20, min_periods=20).mean().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  marzo_test[['sku', 'day', 'promedio_3_active_time']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'promedio_3_active_time', dropna=False).add_prefix('promedio_3_active_time')
subtest4  = marzo_test[['sku', 'day', 'promedio_7_active_time']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'promedio_7_active_time', dropna=False).add_prefix('promedio_7_active_time')
subtest6  = marzo_test[['sku', 'day', 'promedio_15_active_time']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'promedio_15_active_time', dropna=False).add_prefix('promedio_15_active_time')
subtest7  = marzo_test[['sku', 'day', 'promedio_20_active_time']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'promedio_20_active_time', dropna=False).add_prefix('promedio_20_active_time')
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')

del subtest3,subtest4,subtest6, subtest7

#Sumas active time
marzo_test['suma_3_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(3, min_periods=3).sum().reset_index(drop=True)
marzo_test['suma_7_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(7, min_periods=7).sum().reset_index(drop=True)
marzo_test['suma_15_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(15, min_periods=15).sum().reset_index(drop=True)
marzo_test['suma_20_active_time'] = marzo.groupby(['sku'])['minutes_active'].rolling(20, min_periods=20).sum().reset_index(drop=True)
# Pivoteo y mergeo
subtest3 =  marzo_test[['sku', 'day', 'suma_3_active_time']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'suma_3_active_time', dropna=False).add_prefix('suma_3_active_time')
subtest4  = marzo_test[['sku', 'day', 'suma_7_active_time']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'suma_7_active_time', dropna=False).add_prefix('suma_7_active_time')
subtest6  = marzo_test[['sku', 'day', 'suma_15_active_time']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'suma_15_active_time', dropna=False).add_prefix('suma_15_active_time')
subtest7  = marzo_test[['sku', 'day', 'suma_20_active_time']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'suma_20_active_time', dropna=False).add_prefix('suma_20_active_time')
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7

#Sumas sales time
marzo_test['sumas_3'] = marzo.groupby(['sku'])['sold_quantity'].rolling(3, min_periods=3).sum().reset_index(drop=True)
marzo_test['sumas_7'] = marzo.groupby(['sku'])['sold_quantity'].rolling(7, min_periods=7).sum().reset_index(drop=True)
marzo_test['sumas_15'] = marzo.groupby(['sku'])['sold_quantity'].rolling(15, min_periods=15).sum().reset_index(drop=True)
marzo_test['sumas_20'] = marzo.groupby(['sku'])['sold_quantity'].rolling(20, min_periods=20).sum().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  marzo_test[['sku', 'day', 'sumas_3']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'sumas_3', dropna=False).add_prefix('sumas_3')
subtest4  = marzo_test[['sku', 'day', 'sumas_7']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'sumas_7', dropna=False).add_prefix('sumas_7')
subtest6  = marzo_test[['sku', 'day', 'sumas_15']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'sumas_15', dropna=False).add_prefix('sumas_15')
subtest7  = marzo_test[['sku', 'day', 'sumas_20']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'sumas_20', dropna=False).add_prefix('sumas_20')

final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7

final = final.reset_index(drop = False)
final.to_csv('marzo_lgbm_price_norm.csv.gz',index=False, compression="gzip")

