#%% Cargo los datos de febrero nuevamente, los voy a utilizar para comparar si alguna columna de marzo tiene posicion distinta con respecto a febrero
dataset = pd.read_csv('febrero_lgbm_price_norm.csv.gz')  #Cargo el dataset para redes


#Cargo algunos extras
extra = pd.read_csv('features_extras.csv.gz') #Cargo las features extras generadas anteriormente
dataset= dataset.merge(extra, on = 'sku')

#%% Preparo el dataset para armar el predict
test = pd.read_csv('test_data.csv')

#Armo Marzo
final_marzo = pd.read_csv('marzo_lgbm_price_norm.csv.gz')
final_marzo= marzo_final.merge(extra, on = 'sku', how = 'left')
final_marzo = pd.get_dummies(final_marzo) 
final_marzo = final_marzo.merge(test, on='sku')

#%% En caso de que la posicion de las columnas sea distinta, utilizar esta celda
dataset = dataset.drop(['inventory_days'], axis =1 )
cdf_1 = list(dataset.columns.values)
cdf_2 = list(final_marzo.columns.values)

interseccion = [e for e in cdf_2 if e in cdf_1] #a veces me generaba una columna extra vacia cuando hacia el pivot de marzo, por eso este paso
final_marzo = final_marzo[interseccion].copy() 
final_marzo = final_marzo[cdf_1] #Reordeno las columnas igual que en el dataset de febrero

#%% Prediccion sobre datos de marzo
y_pred_mar = pd.DataFrame(clf.predict(final_marzo, num_iteration=clf.best_iteration))  
y_pred_mar['sku']= final_marzo['sku']

submit = submit.set_index('sku')
submit = submit.reindex(test['sku'])
submit = submit.round(4)
submit.to_csv('lgbm.csv.gz',index=False, compression="gzip",  header=False)

