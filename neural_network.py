import numpy as np #v1.16.3
import pandas as pd #v0.24.2
import keras #v2.2.4 Depends on h5py
import tensorflow as tf #TensorFlow v1.13.1
import datetime
import csv

def set_seed (params):
	#If seed = 0 let libraries set a random seed
	if params['seed'] != 0:
		np.random.seed(params['seed'])
		tf.set_random_seed(params['seed'])

def create_nn (params):
	nn = keras.models.Sequential()
	for layer in params['layers']:
		if len(layer) == 3:
			#Three parameter layer is intended for the first one only, with explicit input dimension
			nn.add(keras.layers.Dense(layer[0], kernel_initializer = 'normal', input_dim=layer[2], activation=layer[1]))
		else:
			nn.add(keras.layers.Dense(layer[0], kernel_initializer = 'normal', activation=layer[1]))

	nn.compile(loss='mse', optimizer='adam')

	if 'nn_path' in params:
		nn.save(params['nn_path'], overwrite=True)
	else:
		nn.save('nn/'+str(datetime.datetime.now())+'.h5', overwrite=True)

	return nn

def load_nn (params):
	return keras.models.load_model(params['nn_path'])


def next_move (nn, state, my_pos):
	#Possible mooves
	moves = [np.array([1,0,0,0,0]),np.array([0,1,0,0,0]),np.array([0,0,1,0,0]),np.array([0,0,0,1,0]),np.array([0,0,0,0,1])]
	#Predicted quality of each move
	qs = nn.predict(np.array([np.concatenate((state, my_pos, move), axis=None) for move in moves]))

	best_move = np.argmax(qs)
	return best_move

def load_nl_data(params):
	#Load NL-generated csv
	df = pd.read_csv(params['data_path'], sep=';', header=None, names=['time', 'agent', 'old_pos', 'pos', 'electors_state', 'votes_state', 'next_move', 'previous_votes', 'votes', 'previous_seats', 'seats'])

	#Correctly format each column
	df['time'] = pd.to_numeric(df['time'])
	df['agent'] = pd.to_numeric(df['agent'])
	df['old_pos'] = df['old_pos'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
	df['pos'] = df['pos'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
	df['electors_state'] = df['electors_state'].apply(lambda x: np.fromstring(x[1:-1], sep=' ', dtype=int))
	df['votes_state'] = df['votes_state'].apply(lambda x: np.fromstring(x[1:-1], sep=' ', dtype=int))
	df['next_move'] = pd.to_numeric(df['next_move'])
	df['previous_votes'] = pd.to_numeric(df['previous_votes'])
	df['votes'] = pd.to_numeric(df['votes'])
	df['previous_seats'] = pd.to_numeric(df['previous_seats'])
	df['seats'] = pd.to_numeric(df['seats'])

	return df

def train_nn (params):
	nn = load_nn(params)

	df = load_nl_data (params)

	#Drop unused columns
	df = df.drop(columns=['time', 'agent'])

	if params['target'] == 'seats':
		df = df.drop(columns=['votes', 'previous_votes'])
		target = 'seats'
	else: #'votes'
		df = df.drop(columns=['seats', 'previous_seats'])
		target = 'votes'

	if params['parties_see_electors']: #train with complete infos on electors
		df = df.drop(columns=['votes_state'])
		state = 'electors_state'
	else: #train angainst last election results
		df = df.drop(columns=['electors_state'])
		state = 'votes_state'

	moves = [np.array([1,0,0,0,0]),np.array([0,1,0,0,0]),np.array([0,0,1,0,0]),np.array([0,0,0,1,0]),np.array([0,0,0,0,1])]

	#Input is state (i.e. discretized distribution) + party position + move
	df['learning_input'] = df.apply(lambda x: np.squeeze(np.concatenate((x[state],x['old_pos'], moves[x['next_move']]), axis=0).reshape(-1,1)), axis=1)

	#Target is votes or seats + gamma * prediction of the quality of the next best move
	if params['gamma'] == 0:
		df['learning_target'] = df.apply(lambda x: x[target], axis=1)
	else:
		df['learning_target'] = df.apply(lambda x: x[target] + params['gamma']*np.amax(nn.predict(np.array([np.concatenate((x[state],x['pos'], move), axis=None) for move in moves]))), axis=1)

	#Update weights
	nn.fit(x=np.stack(df['learning_input'].to_numpy()), y=df['learning_target'].to_numpy(), epochs=params['epochs'])

	nn.save(params['nn_path'], overwrite=True)

def get_random_seed():
	#Get a random number which could be used as seed for both NL and NumPy
	return np.random.randint(10000)
