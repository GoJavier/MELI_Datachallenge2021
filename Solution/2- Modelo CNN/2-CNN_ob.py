#Scoring function para validacion de la red
def ranked_probability_score(y_true, y_pred):
    return ((y_true.cumsum(axis=1) - y_pred.cumsum(axis=1))**2).sum(axis=1).mean()

def scoring_function(y_true, y_pred):
    true = y_true.numpy()[:,0]    
    pred = y_pred.numpy()
    true = true.astype(int)
    y_true_one_hot = np.zeros_like(pred, dtype=np.float)    
    y_true_one_hot[range(len(true)), true] = 1
    return ( ranked_probability_score(y_true_one_hot, pred))

#%% Scoring function para la OB
def ranked_probability_score(y_true, y_pred):

    return ((y_true.cumsum(axis=1) - y_pred.cumsum(axis=1))**2).sum(axis=1).mean()
def scoring_function_ob(y_pred, y_true ):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_one_hot = np.zeros_like(y_pred, dtype=np.float)
    y_true_one_hot[range(len(y_true)), y_true] = 1
    return (ranked_probability_score(y_true_one_hot, y_pred))

#%% Custom loss para la red
import tensorflow.keras.backend as K

def loss_rpc(y_true, y_pred):    
    y_true = tf.one_hot(y_true,30)
    y_true = tf.cumsum(y_true, axis = 2)
    y_true = tf.squeeze(y_true)
    y_pred =  tf.cumsum(y_pred, axis = 1)
    resta = tf.subtract(y_true,y_pred) 
    loss = tf.square(resta)
    loss = tf.reduce_sum(loss, 1)
    loss = tf.math.reduce_mean(loss, axis= 0)
    return ( loss)
#%% Grafos de modelos
from tensorflow.keras.layers import *
from tensorflow.keras import Model
def cnn(filtro_1_1, kernel_1_1, filtro_1_2,kernel_1_2, filtro_2_1, kernel_2_1, dense1, batch_size):    
    tf.config.run_functions_eagerly(True)    
    x_sec = Input(shape=(29,6), name = 'secuencias')
    x_feat = Input(shape=(len(x_train_feat[0]),), name = 'features')
    conv1 = Conv1D(int(filtro_1_1),(int(kernel_1_1)),activation="relu",padding="same")(x_sec)
    conv1 = MaxPooling1D(pool_size=(2))(conv1)
    conv1 = Conv1D(int(filtro_1_2),(int(kernel_1_2)),activation="relu",padding="same")(conv1)
    conv1 = MaxPooling1D(pool_size=(2))(conv1)
    conv2 = Conv1D(int(filtro_2_1),(int(kernel_2_1)),activation="relu",padding="same")(x_sec)
    conv2 = MaxPooling1D(pool_size=(4))(conv2)
    join = concatenate([conv1, conv2])
    first_part_output = Flatten()(join)
    merged_model = keras.layers.concatenate([first_part_output, x_feat])
    merged_model = Dense(int(dense1),activation="relu")(merged_model)
    predictions = Dense(30,activation="softmax")(merged_model)
    model = Model(inputs=[x_sec, x_feat], outputs=predictions)    
    optimizer = keras.optimizers.SGD(learning_rate=0.01) 
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_scoring_function', mode='min', patience=30, restore_best_weights= True)
    model.compile(loss=loss_rpc,optimizer=optimizer,metrics=[scoring_function])
    history = model.fit({"secuencias": x_train,"features": x_train_feat},y_train,batch_size=int(batch_size),epochs=4000, validation_data=({"secuencias": x_test,"features": x_test_feat},y_test), callbacks=[callback])
    
    predictions = model.predict(x={"secuencias": x_test,"features": x_test_feat})
    score =1/scoring_function_ob(predictions, y_test)
    print(score)
    return score
#%% Bayes Optimization
#Please install bayes_opt first
from bayes_opt import BayesianOptimization 
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
 #Parameter range setting
bounds_LGB_L1 = {
    'filtro_1_1': (10, 100), 
    'kernel_1_1': (2, 10),   
    'filtro_1_2': (10, 100),
    'kernel_1_2': (2, 10), 
    'filtro_2_1': (10, 100), 
    'kernel_2_1':(2,20),
    'dense1':(200,1000),
    'batch_size':(100,1500),
}
 
 #optimizer
LGB_BO = BayesianOptimization(cnn, bounds_LGB_L1, random_state=7777)
 
init_points = 20 #initial random attempt
n_iter = 150 # optimization attempt
logger = JSONLogger(path="OB_cnn.json")
LGB_BO.subscribe(Events.OPTIMIZATION_STEP, logger)
#Start optimization
LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
 
 #Probe using known reasonable parameters for the probe, may further improve the optimization results
LGB_BO.probe(
    params={'filtro_1_1': 30, 
    'kernel_1_1': 2,   
    'filtro_1_2': 30,
    'kernel_1_2': 2, 
    'filtro_2_1': 30, 
    'kernel_2_1':15,
    'dense1':600,
    'batch_size':500},
    lazy=True, # 
)

LGB_BO.maximize(init_points=0, n_iter=0)
#%% Ejecutar en caso que quieras resetear el kernel

# Guardar el Modelo
model.save('CNN_custom_loss1.tf',save_format='tf')

# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model('CNN_custom_loss1.h5')
