#Genero el dataset de febrero para el approach de boosting. Este approach tiene algunas variables mas incluyendo sumas y promedios de valores pasados

import gc
gc.collect()
import pandas as pd
import seaborn as sns
import numpy as np
#%%   Cargo los datos, Con el dataset de boosting no hice las pruebas de quitarle un dia a marzo y agregarlo a febrero por falta de tiempo
#Se toma febrero y marzo tal como vienen
train = pd.read_parquet(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\train_data.parquet', engine='pyarrow')

#Cambio las variables object a categoricas
for col in ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']:
    train[col] = train[col].astype('category')

    
train['date'] = pd.to_datetime(train['date'])
train['day'] =train.date.dt.day
train['month'] = train.date.dt.month
train['listing_type'] = train['listing_type'].factorize()[0]
train['shipping_logistic_type'] = train['shipping_logistic_type'].factorize()[0]
train['shipping_payment'] = train['shipping_payment'].factorize()[0]
febrero = train.loc[train['month']==2]
marzo = train.loc[train['month']==3]

febrero.to_csv('febrero_limpio.csv.gz',index=False, compression="gzip")
marzo.to_csv('marzo_limpio.csv.gz',index=False, compression="gzip")
#%%  Febrero
febrero = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\febrero_limpio.csv.gz')
#Trabajo mejor el price
febrero = febrero.assign(current_price=febrero.groupby('currency').transform(lambda x: (x - x.min()) / (x.max()- x.min())))

subtest1 =  febrero[['sku', 'day', 'sold_quantity']]
subtest1= subtest1.pivot_table(index = 'sku', columns= 'day', values = 'sold_quantity').add_prefix('sales')

subtest2 =  febrero[['sku', 'day', 'current_price']]
subtest2= subtest2.pivot_table(index = 'sku', columns= 'day', values = 'current_price').add_prefix('price')

subtest3 =  febrero[['sku', 'day', 'minutes_active']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'minutes_active').add_prefix('active_time')


subtest4  = febrero[['sku', 'day', 'listing_type']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'listing_type').add_prefix('listing_type')


subtest6  = febrero[['sku', 'day', 'shipping_logistic_type']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'shipping_logistic_type').add_prefix('shipping_logistic_type')

subtest7  = febrero[['sku', 'day', 'shipping_payment']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'shipping_payment').add_prefix('shipping_payment')

final = pd.merge(subtest1, subtest2, left_index=True, right_index=True )
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)

del subtest1,subtest2,subtest3,subtest4,subtest6, subtest7
#%% Promedios cada 3 dias
febrero_test = febrero.sort_values(['sku','day']).reset_index(drop=True).copy()

febrero_test['promedio_3'] = febrero.groupby(['sku'])['sold_quantity'].rolling(3, min_periods=3).mean().reset_index(drop=True)
febrero_test['promedio_7'] = febrero.groupby(['sku'])['sold_quantity'].rolling(7, min_periods=7).mean().reset_index(drop=True)
febrero_test['promedio_15'] = febrero.groupby(['sku'])['sold_quantity'].rolling(15, min_periods=15).mean().reset_index(drop=True)
febrero_test['promedio_20'] = febrero.groupby(['sku'])['sold_quantity'].rolling(20, min_periods=20).mean().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  febrero_test[['sku', 'day', 'promedio_3']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'promedio_3', dropna=False).add_prefix('promedio_3')
subtest4  = febrero_test[['sku', 'day', 'promedio_7']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'promedio_7', dropna=False).add_prefix('promedio_7')
subtest6  = febrero_test[['sku', 'day', 'promedio_15']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'promedio_15', dropna=False).add_prefix('promedio_15')
subtest7  = febrero_test[['sku', 'day', 'promedio_20']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'promedio_20', dropna=False).add_prefix('promedio_20')

final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7

febrero_test['promedio_3_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(3, min_periods=3).mean().reset_index(drop=True)
febrero_test['promedio_7_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(7, min_periods=7).mean().reset_index(drop=True)
febrero_test['promedio_15_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(15, min_periods=15).mean().reset_index(drop=True)
febrero_test['promedio_20_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(20, min_periods=20).mean().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  febrero_test[['sku', 'day', 'promedio_3_active_time']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'promedio_3_active_time', dropna=False).add_prefix('promedio_3_active_time')
subtest4  = febrero_test[['sku', 'day', 'promedio_7_active_time']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'promedio_7_active_time', dropna=False).add_prefix('promedio_7_active_time')
subtest6  = febrero_test[['sku', 'day', 'promedio_15_active_time']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'promedio_15_active_time', dropna=False).add_prefix('promedio_15_active_time')
subtest7  = febrero_test[['sku', 'day', 'promedio_20_active_time']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'promedio_20_active_time', dropna=False).add_prefix('promedio_20_active_time')
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')

del subtest3,subtest4,subtest6, subtest7

#Sumas active time
febrero_test['suma_3_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(3, min_periods=3).sum().reset_index(drop=True)
febrero_test['suma_7_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(7, min_periods=7).sum().reset_index(drop=True)
febrero_test['suma_15_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(15, min_periods=15).sum().reset_index(drop=True)
febrero_test['suma_20_active_time'] = febrero.groupby(['sku'])['minutes_active'].rolling(20, min_periods=20).sum().reset_index(drop=True)
# Pivoteo y mergeo
subtest3 =  febrero_test[['sku', 'day', 'suma_3_active_time']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'suma_3_active_time', dropna=False).add_prefix('suma_3_active_time')
subtest4  = febrero_test[['sku', 'day', 'suma_7_active_time']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'suma_7_active_time', dropna=False).add_prefix('suma_7_active_time')
subtest6  = febrero_test[['sku', 'day', 'suma_15_active_time']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'suma_15_active_time', dropna=False).add_prefix('suma_15_active_time')
subtest7  = febrero_test[['sku', 'day', 'suma_20_active_time']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'suma_20_active_time', dropna=False).add_prefix('suma_20_active_time')
final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7

#Sumas sales time
febrero_test['sumas_3'] = febrero.groupby(['sku'])['sold_quantity'].rolling(3, min_periods=3).sum().reset_index(drop=True)
febrero_test['sumas_7'] = febrero.groupby(['sku'])['sold_quantity'].rolling(7, min_periods=7).sum().reset_index(drop=True)
febrero_test['sumas_15'] = febrero.groupby(['sku'])['sold_quantity'].rolling(15, min_periods=15).sum().reset_index(drop=True)
febrero_test['sumas_20'] = febrero.groupby(['sku'])['sold_quantity'].rolling(20, min_periods=20).sum().reset_index(drop=True)

# Pivoteo y mergeo
subtest3 =  febrero_test[['sku', 'day', 'sumas_3']]
subtest3= subtest3.pivot_table(index = 'sku', columns= 'day', values = 'sumas_3', dropna=False).add_prefix('sumas_3')
subtest4  = febrero_test[['sku', 'day', 'sumas_7']]
subtest4= subtest4.pivot_table(index = 'sku', columns= 'day', values = 'sumas_7', dropna=False).add_prefix('sumas_7')
subtest6  = febrero_test[['sku', 'day', 'sumas_15']]
subtest6= subtest6.pivot_table(index = 'sku', columns= 'day', values = 'sumas_15', dropna=False).add_prefix('sumas_15')
subtest7  = febrero_test[['sku', 'day', 'sumas_20']]
subtest7= subtest7.pivot_table(index = 'sku', columns= 'day', values = 'sumas_20', dropna=False).add_prefix('sumas_20')

final = pd.merge(final, subtest3, left_index=True, right_index=True)
final = pd.merge(final, subtest4, left_index=True, right_index=True)
final = pd.merge(final, subtest6, left_index=True, right_index=True)
final = pd.merge(final, subtest7, left_index=True, right_index=True)
final = final.dropna(axis=1, how='all')
del subtest3,subtest4,subtest6, subtest7


#%% Creo las variables target
marzo = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\marzo_limpio.csv.gz')

from tqdm import tqdm
def create_validation_set(dataset):
    np.random.seed(42)
    print('Sorting records...')
    temp_pd = dataset.loc[:, ['sku', 'date', 'sold_quantity']].sort_values(['sku', 'date'])
    print ('Grouping quantity')
    temp_dict = temp_pd.groupby('sku').agg({'sold_quantity':lambda x: [i for i in x]})['sold_quantity'].to_dict()
    result = []
    for idx, list_quantity in tqdm(temp_dict.items(), desc = 'Making targets ...'):
        cumsum = np.array(list_quantity).cumsum()

        stock_target = 0
        if cumsum[-1] > 0:
            while stock_target==0:
                stock_target = np.random.choice(cumsum)

            
            day_to_stockout = np.argwhere(cumsum==stock_target).min()+1
            result.append({'sku':idx, 'target_stock':stock_target, 'inventory_days': day_to_stockout})
    return result
marzo_final = create_validation_set(marzo.loc[marzo['day']<=30].reset_index(drop=True))
    
#%% Adjunto las variables a predecir

marzo_target = pd.DataFrame(marzo_final)

final_test = final.copy()
marzo_target_test = marzo_target.copy()

dataset = final_test.merge(marzo_target_test, on='sku')
dataset['inventory_days']=dataset['inventory_days']-1 

dataset.to_csv('febrero_lgbm_price_norm.csv.gz',index=False, compression="gzip")

