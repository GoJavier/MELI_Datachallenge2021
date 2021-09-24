#La predicción final es un promedio del resultado obtenido por la red y por el boosting

modelo_cnn = pd.read_csv('cnn.csv.gz', header = None)
modelo_lgb = pd.read_csv('lgbm.csv.gz', header = None)

#%%
promedio = (modelo_cnn+modelo_lgb)/2
promedio = promedio.round(4)
promedio.to_csv('modelo_promedio.csv.gz',index=False, compression="gzip",float_format='%.4f',  header=False)
