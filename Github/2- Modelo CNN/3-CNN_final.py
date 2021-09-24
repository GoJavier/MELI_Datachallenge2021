def ranked_probability_score(y_true, y_pred):

    return ((y_true.cumsum(axis=1) - y_pred.cumsum(axis=1))**2).sum(axis=1).mean()

def scoring_function(y_true, y_pred):
    true = y_true.numpy()[:,0]
    
    pred = y_pred.numpy()
    true = true.astype(int)
    y_true_one_hot = np.zeros_like(pred, dtype=np.float)

    
    y_true_one_hot[range(len(true)), true] = 1
    return ( ranked_probability_score(y_true_one_hot, pred))
#%% Custom loss
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


#%% Grafos de modelos, utilizo los valores conseguidos en la optimizacion bayesiana (esto se utilizó en la competencia)
from tensorflow.keras import Model
from tensorflow.keras.layers import *
tf.config.run_functions_eagerly(True)

x_sec = Input(shape=(29,6), name = 'secuencias')
x_feat = Input(shape=(len(x_train_feat[0]),), name = 'features')
conv1 = Conv1D(87,(6),activation="relu",padding="same")(x_sec)
conv1 = MaxPooling1D(pool_size=(2))(conv1)
conv1 = Conv1D(63,(3),activation="relu",padding="same")(conv1)
conv1 = MaxPooling1D(pool_size=(2))(conv1)
conv2 = Conv1D(13,(11),activation="relu",padding="same")(x_sec)
conv2 = MaxPooling1D(pool_size=(4))(conv2)

join = concatenate([conv1, conv2])
first_part_output = Flatten()(join)
merged_model = keras.layers.concatenate([first_part_output, x_feat])
merged_model = Dense(991,activation="relu")(merged_model)
predictions = Dense(30,activation="softmax")(merged_model)
model = Model(inputs=[x_sec, x_feat], outputs=predictions)
model.summary()

optimizer = keras.optimizers.SGD(learning_rate=0.01) 
callback = tf.keras.callbacks.EarlyStopping(monitor='val_scoring_function', mode='min', patience=30, restore_best_weights= True)
model.compile(loss=loss_rpc,optimizer=optimizer,metrics=[scoring_function])
history = model.fit({"secuencias": x_train,"features": x_train_feat},y_train,batch_size=1497,epochs=45)
