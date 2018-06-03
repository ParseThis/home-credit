
import sys


from keras.models import Model
from keras.layers import Input, Dense, Conv1D, concatenate, Flatten
from keras.optimizers import Adam

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import logging

import pandas as pd
import numpy as np

import gc

SEED = 42
TEST_SIZE = 0.1




INPUT = '../data/application_train.csv'
TEST_INPUT = '../data/application_test.csv'

def get_data():
	
	data = pd.read_csv(INPUT)
	test = pd.read_csv(TEST_INPUT)

	categorical_feats = data.iloc[:, 1:-1].select_dtypes('O').columns
	for f_ in categorical_feats:
	    data[f_], indexer = pd.factorize(data[f_])
	    test[f_] = indexer.get_indexer(test[f_])

	y = data.TARGET
	return data.iloc[:, 1:-1], y, test


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)



if __name__ == "__main__":

	import logging
	logging.basicConfig(filename='example.log',level=logging.DEBUG)
	d1_neurons = 512
	d2_neurons = 512
	num_classes = 1
	validation_size = 0.1
	batch_size=32
	num_epochs=5
	
	data, y, test= get_data()
	# basic fillna
	data = data.fillna(0)
	test = test.fillna(0)


	X_train, X_test, y_train, y_test = train_test_split(
		data, y, random_state=SEED, test_size=TEST_SIZE)


	# pipe = Pipeline([
	# 			('scale', StandardScaler())])
	# pipe.fit(data)
	# X = pipe.transform(data)
	# test_ = pipe.transform(test.iloc[:, 1:])


	inp = Input(shape=(X_train.shape[1],))
	d1 = Dense(d1_neurons)(inp)
	d2  = Dense(d2_neurons)(d1)
	out = Dense(num_classes, activation='sigmoid')(d2)
	model = Model(inputs =[inp], outputs=[out])



	kf = KFold(n_splits=10, random_state=42)
	oof_preds = np.zeros(X_train.shape[0])

	inp = Input(shape=(X_train.shape[1],))
	d1 = Dense(d1_neurons)(inp)
	d2  = Dense(d2_neurons)(d1)
	out = Dense(num_classes, activation='sigmoid')(d2)
	model = Model(inputs =[inp], outputs=[out])
	opt = Adam(0.001)
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer=opt)




	for n_fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
	    trn_x, trn_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
	    val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]
	   	

	    model.fit(trn_x, trn_y, batch_size=batch_size, epochs=num_epochs)
	    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, model.predict(val_x))))
	    del trn_x, trn_y, val_x, val_y
	    gc.collect()



	predictions = model.predict(X_test)
	score = roc_auc_score(y_test, predictions)

	print('SCORE: %d', score)
	test['TARGET'] = model.predict(test_)
	# test[['SK_ID_CURR', 'TARGET']]\
	# 	.to_csv('../data/basic_ff_submission.csv', index=False, float_format='%.8f')
	print('DONE')





	