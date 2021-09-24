import gc
gc.collect()
import pandas as pd
import numpy as np
from tensorflow import keras
import pandas as pd
import numpy as np 


#%% Cargo datos
dataset = pd.read_csv('febrero_lgbm_price_normx29_redes.csv.gz')  #Cargo el dataset para redes
extra = pd.read_csv('features_extras.csv.gz') #Cargo las features extras generadas anteriormente
#Merge metadata con dataset febrero
dataset= dataset.merge(extra, on = 'sku')

del extra
#%% Normalizacion por grupos
dataset = dataset.fillna(0) #Para redes no tengo que tener Nas, con mas tiempo podria haberse probado algún método de imputacion

#Saco lo que no quiero normalizar
dataset_inventory = dataset['inventory_days']
dataset = dataset.drop(['inventory_days'], axis =1)

cols = dataset.columns.tolist()
to_keep = cols[30:59]
dataset_prices = dataset[to_keep] #Los precios ya los tengo normalizado por tipo de moneda, prefiero retirarlos de esta normalizacion

dataset = dataset.loc[:, ~dataset.columns.isin(to_keep)]

dataset_sku = dataset['sku']
dataset_target = dataset['target_stock']
dataset = dataset.drop(['sku', 'target_stock'], axis =1)
dataset = pd.get_dummies(dataset)
dataset_normalized = dataset.copy()  
for col in dataset_normalized.columns:  
    q=np.quantile(dataset_normalized[col],0.975)
    dataset_normalized[col] = ((dataset_normalized[col]*2)/q).clip(upper=3)

dataset_normalized['sku']= dataset_sku
dataset_normalized['target_stock']= dataset_target


dataset_normalized = dataset_normalized.merge(dataset_prices, left_index = True, right_index = True)
dataset = dataset_normalized.copy()
dataset = dataset[cols]
dataset['inventory_days']= dataset_inventory

del dataset_sku
#%%  Ajuste de variables
#Decidi no normalizar el target stock para que la red le de mas relevancia a la hora de optimizar, igualmente divido un poco los valores para que no se sature
dataset['target_stock'] = dataset['target_stock'].div(6)

#%% Creo las variables a entrenar

train=dataset.sample(frac=0.8,random_state=7777) 
test=dataset.drop(train.index)


#Armo los channels para la CNN
#Sales
x_train = train.iloc[:,1:30]
x_train = x_train.to_numpy()
x_train = x_train.reshape((-1, 29, 1))
#Price
x_train_price = train.iloc[:,30:59]
x_train_price = x_train_price.to_numpy()
x_train = np.dstack((x_train, x_train_price))
#Active Time
x_train_activetime = train.iloc[:,59:88]
x_train_activetime = x_train_activetime.to_numpy()
x_train = np.dstack((x_train, x_train_activetime))
# #Listing type
x_train_listing_type = train.iloc[:,88:117]
x_train_listing_type = x_train_listing_type.to_numpy()
x_train = np.dstack((x_train, x_train_listing_type))
#Shipping
x_train_shipping = train.iloc[:,117:146]
x_train_shipping = x_train_shipping.to_numpy()
x_train = np.dstack((x_train, x_train_shipping))

#Shipping
x_train_shipping_pay = train.iloc[:,146:175]
x_train_shipping_pay = x_train_shipping_pay.to_numpy()
x_train = np.dstack((x_train, x_train_shipping_pay))

#Features para la parte fully connected de la red y target a predecir
x_train_feat = train.iloc[:,-9:-1]

x_train_feat['sku'] = train['sku'].div(100000)
x_train_feat = x_train_feat.to_numpy()
y_train = train['inventory_days'].to_numpy()

#Sales
x_test = test.iloc[:,1:30]
x_test = x_test.to_numpy()
x_test = x_test.reshape((-1, 29, 1))

#Price
x_test_price = test.iloc[:,30:59]
x_test_price = x_test_price.to_numpy()
x_test = np.dstack((x_test, x_test_price))

#Active Time
x_test_activetime = test.iloc[:,59:88]
x_test_activetime = x_test_activetime.to_numpy()
x_test = np.dstack((x_test, x_test_activetime))
#Listing type
x_test_listing_type = test.iloc[:,88:117]
x_test_listing_type = x_test_listing_type.to_numpy()
x_test = np.dstack((x_test, x_test_listing_type))
#Shipping
x_test_shipping = test.iloc[:,117:146]
x_test_shipping = x_test_shipping.to_numpy()
x_test = np.dstack((x_test, x_test_shipping))

#Shipping
x_test_shipping_pay = test.iloc[:,146:175]
x_test_shipping_pay = x_test_shipping_pay.to_numpy()
x_test = np.dstack((x_test, x_test_shipping_pay))

#Features y target
x_test_feat = test.iloc[:,-9:-1]


x_test_feat['sku'] = test['sku'].div(100000)
x_test_feat = x_test_feat.to_numpy()
y_test = test['inventory_days'].to_numpy()
