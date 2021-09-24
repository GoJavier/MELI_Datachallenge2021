#Ajusto el dataset de marzo para que sea igual al de febrero (features y normalizaciones) y luego predigo
#%%
from tensorflow import keras
import pandas as pd
import numpy as np
test = pd.read_csv('test_data.csv')
final_marzo = pd.read_csv('marzo_lgbm_price_normx30_redes.csv.gz')
extra = pd.read_csv('features_extras.csv.gz')
final_marzo= final_marzo.merge(extra, on = 'sku')
final_marzo= final_marzo.merge(test, on = 'sku')
final_marzo = final_marzo.fillna(0)
del extra

#%% Normalizacion por grupos
final_marzo_stock = final_marzo['target_stock']
cols2 = final_marzo.columns.tolist()
to_keep = cols2[30:59]
dataset_prices = final_marzo[to_keep]
final_marzo = final_marzo.loc[:, ~final_marzo.columns.isin(to_keep)]

dataset_sku = final_marzo['sku']
final_marzo = final_marzo.drop(['sku', 'target_stock'], axis =1)
final_marzo = pd.get_dummies(final_marzo)
dataset_normalized = final_marzo.copy()  


for col in dataset_normalized.columns:  #Otro approach podria ser por variable total y no por dia
    
    q = np.quantile(dataset_normalized[col], 0.975)  #Si pongo un poco menos me queda en 0 varias categorias
    dataset_normalized [col] = ((dataset_normalized [col]*2)/q).clip(upper= 3)


dataset_normalized['sku']= dataset_sku
dataset_normalized['target_stock']= final_marzo_stock
dataset_normalized = dataset_normalized.merge(dataset_prices, left_index = True, right_index = True)

final_marzo = dataset_normalized.copy()
final_marzo = final_marzo[cols]
dataset_normalized = dataset_normalized[cols]
del q, col

#%%  Ajusto target stock (tal como se hizo con el dataset de training)

dataset_normalized['target_stock'] = dataset_normalized['target_stock'].div(6)

#%% Armo el input
#Armo los channels
#Sales
marzo = dataset_normalized.iloc[:,1:30]
marzo = marzo.to_numpy()
marzo = marzo.reshape((-1, 29, 1))
#Price
marzo_price = dataset_normalized.iloc[:,30:59]
marzo_price = marzo_price.to_numpy()
marzo = np.dstack((marzo, marzo_price))
#Active Time
marzo_activetime = dataset_normalized.iloc[:,59:88]
marzo_activetime = marzo_activetime.to_numpy()
marzo = np.dstack((marzo, marzo_activetime))
#Listing type
marzo_listing_type = dataset_normalized.iloc[:,88:117]
marzo_listing_type = marzo_listing_type.to_numpy()
marzo = np.dstack((marzo, marzo_listing_type))
#Shipping
marzo_shipping = dataset_normalized.iloc[:,117:146]
marzo_shipping = marzo_shipping.to_numpy()
marzo = np.dstack((marzo, marzo_shipping))

#Shipping
marzo_shipping_pay = dataset_normalized.iloc[:,146:175]
marzo_shipping_pay = marzo_shipping_pay.to_numpy()
marzo = np.dstack((marzo, marzo_shipping_pay))

#Features
marzo_feat = dataset_normalized.iloc[:,-8:]
marzo_feat['sku'] = dataset_normalized['sku'].div(100000)
marzo_feat = marzo_feat.to_numpy()
#%% Realizo la prediccion sobre marzo
y_pred = model.predict(x={"secuencias": marzo,"features": marzo_feat})
#%% Exporto los resultados
y_pred_mar = pd.DataFrame(y_pred)  
y_pred_mar['sku']= final_marzo['sku']
submit= y_pred_mar.copy()




submit = submit.set_index('sku')
submit = submit.reindex(test['sku'])
submit = submit.round(4)
submit.to_csv('cnn.csv.gz',index=False, compression="gzip",float_format='%.4f',  header=False)

