#Preparo el dataset para utilizar en el approach de redes neuronales, febrero será training y marzo validacion

import gc
gc.collect()
import pandas as pd
import seaborn as sns
import numpy as np
#%%   Cargo los datos
train = pd.read_parquet(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\train_data.parquet', engine='pyarrow')

#Cambio las variables object a categoricas
for col in ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']:
    train[col] = train[col].astype('category')

#Genero informacion con las fechas y trabajo las categoricas
train['date'] = pd.to_datetime(train['date'])
train['day'] =train.date.dt.day
train['month'] = train.date.dt.month
train['listing_type'] = train['listing_type'].factorize()[0]
train['shipping_logistic_type'] = train['shipping_logistic_type'].factorize()[0]
train['shipping_payment'] = train['shipping_payment'].factorize()[0]
#Como voy a utilizar febrero para entrenar y solo tiene 28 dias y yo necesito predecir los 30 dias siguientes, le quito el 1 a marzo y se lo agrego a febrero
train.loc[(train['month']==3) & (train['day']==1), 'day'] = 60 
train.loc[(train['month']==3) & (train['day']==60), 'month'] = 2 
train.loc[(train['month']==2) & (train['day']==60), 'day'] = 29 

febrero = train.loc[train['month']==2]
marzo = train.loc[train['month']==3]

febrero.to_csv('febrero_limpiox29.csv.gz',index=False, compression="gzip")
marzo.to_csv('marzo_limpiox30.csv.gz',index=False, compression="gzip")
#%%  Febrero
febrero = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\febrero_limpiox29.csv.gz')
#Normalizo las variables de interes en base al tipo de moneda, el resto de las variables seran normalizadas mas adelante
febrero = febrero.assign(current_price=febrero.groupby('currency').transform(lambda x: (x - x.min()) / (x.max()- x.min())))
febrero = febrero.assign(sold_quantity=febrero.groupby('currency').transform(lambda x: (x - x.min()) / (x.max()- x.min())))

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

#%% Creo las variables target (script extraida del workshop de MELI Data Challenge 2021)
marzo = pd.read_csv(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Project MELI\Dataset_limpios\marzo_limpiox30.csv.gz')

from tqdm import tqdm
def create_validation_set(dataset):
    np.random.seed(42)   #No es el seed utilizado originalmente, ese lo perdi.
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
marzo_final = create_validation_set(marzo.loc[marzo['day']<=31].reset_index(drop=True))
    
#%% Adjunto las variables a predecir

marzo_target = pd.DataFrame(marzo_final)

final_test = final.copy()
marzo_target_test = marzo_target.copy()

dataset = final_test.merge(marzo_target_test, on='sku')
dataset['inventory_days']=dataset['inventory_days']-1 

dataset.to_csv('febrero_lgbm_price_normx29_redes.csv.gz',index=False, compression="gzip")

