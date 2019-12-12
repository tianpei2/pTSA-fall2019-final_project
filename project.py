import pandas as pd 
import numpy as np 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 

################ Configuration #################
tf.random.set_seed(101)
np.random.seed(101)
TIME_STEPS = 21
BATCH_SIZE = 63
LR = 0.0001
NUM_OF_EPOCH = 20
NUM_OF_NEURONS = 100
DROPOUT_RATE = 0.4
NUM_NEURONS_FC = 20
OPTIMIZER = 'RMSprop'
ACTIVATION = 'relu'

def build_timeseries(mat, TIME_STEPS, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    return x, y

def trim_dataset(mat,BATCH_SIZE):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % BATCH_SIZE
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat
 
def LSTM_model(x_train, y_train, x_val, y_val, TIME_STEPS, BATCH_SIZE, NUM_OF_EPOCH, NUM_OF_NEURONS, DROPOUT_RATE, NUM_NEURONS_FC, LR, ACTIVATION):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    features = x_train.shape[2]
    lstm_model.add(LSTM(NUM_OF_NEURONS, batch_input_shape=(BATCH_SIZE, TIME_STEPS, features),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(DROPOUT_RATE))
    lstm_model.add(Dense(NUM_NEURONS_FC, activation=ACTIVATION))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=LR)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    res = lstm_model.fit(x_train, y_train, epochs = NUM_OF_EPOCH, verbose = 0, batch_size = BATCH_SIZE, validation_data = (x_val, y_val))
    return res

K_FOLD = 5 # k = 5
def train_model(x_t, y_t, TIME_STEPS, BATCH_SIZE, NUM_OF_EPOCH, NUM_OF_NEURONS, DROPOUT_RATE, NUM_NEURONS_FC, LR, ACTIVATION):
	######## k-Fold validation #######
	skf = KFold(n_splits = K_FOLD)
	ret = []
	for t_ind, v_ind in skf.split(x_t, y_t):
		x_train, x_val = x_t[t_ind], x_t[v_ind]
		y_train, y_val = y_t[t_ind], y_t[v_ind]
		x_train, x_val = trim_dataset(x_train, BATCH_SIZE), trim_dataset(x_val, BATCH_SIZE)
		y_train, y_val = trim_dataset(y_train, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE)
		res = LSTM_model(x_train, y_train, x_val, y_val, TIME_STEPS, BATCH_SIZE, NUM_OF_EPOCH, NUM_OF_NEURONS, DROPOUT_RATE, NUM_NEURONS_FC, LR, ACTIVATION)
		ret.append(res)
	return ret


df_brentOil = pd.read_csv('data.csv')
def read_Data(df, TIME_STEPS, TRAIN_SIZE, BATCH_SIZE):
	# df_train, df_test = df[:TRAIN_SIZE], df[TRAIN_SIZE-TIME_STEPS:TRAIN_SIZE+BATCH_SIZE]
	df_train, df_test = train_test_split(df, test_size=0.2, shuffle = False)
	listOfColumns = df.columns.values[1:]
	x_train = df_train.loc[:, listOfColumns].values
	x_test = df_test.loc[:, listOfColumns].values
	min_max_scaler =MinMaxScaler()
	x_train = min_max_scaler.fit_transform(x_train)
	x_test = min_max_scaler.fit_transform(df_test.loc[:, listOfColumns])
	x_t, y_t = build_timeseries(x_train, TIME_STEPS, 3)
	x_test, y_test = build_timeseries(x_test, TIME_STEPS, 3)

	return x_t, y_t, x_test, y_test

def format(data):
	# res = np.zeros((len(data), len(data[0].history['loss'])))
	loss = []
	val_loss = []
	for i in range(len(data)):
		loss.append(data[i].history['loss'])
		val_loss.append(data[i].history['val_loss'])
	loss = np.array(loss)
	val_loss = np.array(val_loss)
	return loss, val_loss


####################### MAIN ########################
TRAIN_SIZE = 504
from matplotlib import pyplot as plt
x_t, y_t, x_test, y_test = read_Data(df_brentOil, TIME_STEPS, TRAIN_SIZE, BATCH_SIZE)
for ACTIVATION in ['relu', 'tanh', 'sigmoid']:
	ret = train_model(x_t, y_t, TIME_STEPS, BATCH_SIZE, NUM_OF_EPOCH, NUM_OF_NEURONS, DROPOUT_RATE, NUM_NEURONS_FC, LR, ACTIVATION)
	loss, val_loss = format(ret)
	plt.figure()
	plt.boxplot(loss)
	plt.title('Training loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.savefig('activation/Training_loss: {}'.format(ACTIVATION))
	plt.close()

	plt.figure()
	plt.boxplot(val_loss)
	plt.title('Validation loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.savefig('activation/Validation_loss: {}'.format(ACTIVATION))
	plt.close()

