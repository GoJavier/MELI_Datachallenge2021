import lightgbm as lgb
import pandas as pd
import numpy as np
#%% Cargo los datos de boosting
dataset = pd.read_csv('febrero_lgbm_price_norm.csv.gz')  #Cargo el dataset para redes
extra = pd.read_csv('features_extras.csv.gz') #Cargo las features extras generadas anteriormente

dataset= dataset.merge(extra, on = 'sku')

del extra

#%%
#Preparo los datasets

train=dataset.sample(frac=0.8,random_state=7777)  #Este random no es el usado orignalmente, ese no lo pude recuperar
test=dataset.drop(train.index)

x_train = train.drop(['inventory_days'], axis = 1)
X_train = train.drop(['inventory_days'], axis = 1)
y_train = train['inventory_days']

X_test = test.drop(['inventory_days'], axis = 1)
y_test = test['inventory_days']

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test,y_test, reference=lgb_train)

#%% Scoring lightgbm
def ranked_probability_score(y_true, y_pred):
    return ((y_true.cumsum(axis=1) - y_pred.cumsum(axis=1))**2).sum(axis=1).mean()

def scoring_function(y_pred, y_true ):
    y_true = np.array(y_true.get_label().astype(int))
    y_pred = np.reshape(y_pred, (len(np.unique(y_true)), -1))    
    y_pred = np.transpose(y_pred)    
    y_true_one_hot = np.zeros_like(y_pred, dtype=np.float)

    y_true_one_hot[range(len(y_true)), y_true] = 1
    return ('Score', ranked_probability_score(y_true_one_hot, y_pred), False)
#%% Scoring function OB
def ranked_probability_score(y_true, y_pred):

    return ((y_true.cumsum(axis=1) - y_pred.cumsum(axis=1))**2).sum(axis=1).mean()
def scoring_function_ob(y_pred, y_true ):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_one_hot = np.zeros_like(y_pred, dtype=np.float)

    y_true_one_hot[range(len(y_true)), y_true] = 1
    return (ranked_probability_score(y_true_one_hot, y_pred))


#%% Funcion lgbm para la opt bayesiana sin folds
def LGB_L1_bayesian(num_leaves, learning_rate, feature_fraction,
                lambda_l1, lambda_l2, max_depth, bagging_fraction, bagging_freq):
    
    
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    bagging_freq = int(bagging_freq)
    
    assert type(num_leaves) == int
    assert type(max_depth) == int
    assert type(bagging_freq) == int

    
    param = {
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'max_depth': max_depth,
        'save_binary': True, 
        'seed': 1337,
        'max_bin': 30,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'multiclass',
        'metric': 'custom',
        'verbose': -1,
        'num_threads': 20,
        'num_class': 30,
        'boost_from_average': False
 
    }    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test,y_test, reference=lgb_train)
    
    
    num_round = 10000
    clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_eval], early_stopping_rounds = int((1/learning_rate)+20),verbose_eval=10, feval =scoring_function )
    
    predictions = clf.predict(X_test, num_iteration=clf.best_iteration)   
    
    score =1/scoring_function_ob(predictions, y_test)

    return score
#%% Bayes Optimization

from bayes_opt import BayesianOptimization 
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

bounds_LGB_L1 = {
    'num_leaves': (20, 50), 
    'learning_rate': (0.01, 0.1),   
    'feature_fraction': (0.1, 1),
    'lambda_l1': (0, 20.0), 
    'lambda_l2': (0, 20.0), 
    'max_depth':(3,15),
    'bagging_fraction':(0.2,1),
    'bagging_freq':(1,10),
}
 

LGB_BO = BayesianOptimization(LGB_L1_bayesian, bounds_LGB_L1, random_state=13)
 
init_points = 20 
n_iter = 120 
logger = JSONLogger(path="Corrida_4?lgb_muchascols.json")
LGB_BO.subscribe(Events.OPTIMIZATION_STEP, logger)

LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
 
LGB_BO.probe(
    params={'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'lambda_l2': 5, 
            'lambda_l1': 0,
            'learning_rate': 0.05,
            'max_depth': 5, 
            'num_leaves': 31},
    lazy=True, 
)
LGB_BO.maximize(init_points=0, n_iter=0)
#%% Mejores Parametros

#En caso que se haya corrido la ob en el mismo kernel utilizar este comando. Sino ejecutar el de la siguiente celda que fue el utilizado en la competencia
best_params = LGB_BO.max['params']
#%% Estos fueron los parámetros utilizados para la competencia
best_params={'feature_fraction': 0.84,
            'bagging_fraction': 0.822,
            'bagging_freq': 3.13,
            'lambda_l2': 9.72, 
            'lambda_l1': 5.65,
            'learning_rate': 0.01,
            'max_depth': 10.30, 
            'num_leaves': 43.26}

#%% 
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['bagging_freq'] = int(best_params['bagging_freq'])
    
# assert type(num_leaves) == int
# assert type(max_depth) == int
# assert type(bagging_freq) == int

    
param1= {'save_binary': True, 
        'seed': 1337,
        'max_bin': 30,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'multiclass',
        'verbose': -1,
        'metric': 'custom',
        'num_threads': 20,
        'num_class': 30,
        'boost_from_average': False}
    

param = {**best_params, **param1}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test,y_test, reference=lgb_train)

num_round = 10000
clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_eval], early_stopping_rounds = int((1/best_params['learning_rate'])+20),verbose_eval=10, feval =scoring_function )
#%% Guardo el modelo
clf.save_model('modelo_overfit_lgbm.txt')

clf = lgb.Booster(model_file=r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\Optimizacion Bayesiana\rolling_model.txt')
