import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import gc
from lightgbm import LGBMClassifier
import xgboost as xgb
import functools


import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)



BUILDING_FEATURES = """ APARTMENTS_AVG
		BASEMENTAREA_AVG
		YEARS_BEGINEXPLUATATION_AVG
		YEARS_BUILD_AVG
		COMMONAREA_AVG
		ELEVATORS_AVG
		ENTRANCES_AVG
		FLOORSMAX_AVG
		FLOORSMIN_AVG
		LANDAREA_AVG
		LIVINGAPARTMENTS_AVG
		LIVINGAREA_AVG
		NONLIVINGAPARTMENTS_AVG
		NONLIVINGAREA_AVG
		APARTMENTS_MODE
		BASEMENTAREA_MODE
		YEARS_BEGINEXPLUATATION_MODE
		YEARS_BUILD_MODE
		COMMONAREA_MODE
		ELEVATORS_MODE
		ENTRANCES_MODE
		FLOORSMAX_MODE
		FLOORSMIN_MODE
		LANDAREA_MODE
		LIVINGAPARTMENTS_MODE
		LIVINGAREA_MODE
		NONLIVINGAPARTMENTS_MODE
		NONLIVINGAREA_MODE
		APARTMENTS_MEDI
		BASEMENTAREA_MEDI
		YEARS_BEGINEXPLUATATION_MEDI
		YEARS_BUILD_MEDI
		COMMONAREA_MEDI
		ELEVATORS_MEDI
		ENTRANCES_MEDI
		FLOORSMAX_MEDI
		FLOORSMIN_MEDI
		LANDAREA_MEDI
		LIVINGAPARTMENTS_MEDI
		LIVINGAREA_MEDI
		NONLIVINGAPARTMENTS_MEDI
		NONLIVINGAREA_MEDI
		FONDKAPREMONT_MODE
		HOUSETYPE_MODE
		TOTALAREA_MODE
		WALLSMATERIAL_MODE
		EMERGENCYSTATE_MODE""".split('\n') 

BUILDING_FEATURES =  list(map(str.strip, BUILDING_FEATURES))
	# contact information
CONTACT_FEATURES = """ FLAG_MOBIL
	FLAG_EMP_PHONE  
	FLAG_WORK_PHONE 
	FLAG_CONT_MOBILE
	FLAG_PHONE      
	FLAG_EMAIL """.split('\n')
CONTACT_FEATURES = list(map(str.strip, 	CONTACT_FEATURES))



CENTER_FEATURES = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE']

SEED = 42
TEST_SIZE = 0.1
FOLDS  = 5



INPUT = '../data/application_train.csv'
TEST_INPUT = '../data/application_test.csv'
BUREAU = '../data/bureau.csv'
BUREAU_BAL = '../data/bureau_balance.csv'
POS_CAHS_BAL = '../data/POS_CASH_balance.csv'
CREDIT_CARD_BAL = '../data/credit_card_balance.csv'
PREV_APP = '../data/previous_application.csv'
INST_PAYMENT = '../data/installments_payments.csv'

def get_bureau_features(data):

	bureau = pd.read_csv(BUREAU)
	active_credit_group = bureau[bureau.CREDIT_ACTIVE == 'Active'].groupby('SK_ID_CURR')
	appl_bureau_group = bureau.groupby('SK_ID_CURR')

	# active_bureau_loans = bureau.filter(lambd.groupby()

	num_active_credit = active_credit_group.apply(len)
	num_credit_bureau_loans = appl_bureau_group.apply(len)
	
	return 


def get_app_complete_features(data):
		# building features
	building_data_mask = data[BUILDING_FEATURES].isnull().apply(sum, axis=1)
	# complete building information
	return building_data_mask > 0


def dist_from_center(data, feature):

	# distance from mean AMT_CREDIT
	center_features = {}
	mn_ = data[feature].mean()
	mdn_ = data[feature].median()

	aggs = (( mn_, 'mean') , (mdn_, 'median'))

	def diffs(agg_quantity, agg_type='mean'):
		diff = data[feature] - agg_quantity
		abs_diff = diff.apply(abs)
		center_features['abs_diff_from_' + agg_type + '_'  + feature] = abs_diff
		return center_features


	[diffs(ag, t) for ag, t in aggs]
	return center_features


def merge_grouping_in_sk_id(data, feature, feature_name):
	"feature is a seriees"
	new_ = feature.reset_index()
	new_.columns = ['SK_ID_CURR', feature_name]

	d_ = data.merge(new_, how='left', on='SK_ID_CURR')
	d_[feature_name] = d_[feature_name].fillna(0) # if data doesnt' exit assume zero
	return d_

def get_features(train, test):



	bureau =pd.read_csv(BUREAU)
	train['app_complete_building'] = get_app_complete_features(train)
	test['app_complete_building'] = get_app_complete_features(test)

	# BUREAU FEATURES

	active_credit_group = bureau[bureau.CREDIT_ACTIVE == 'Active'].groupby('SK_ID_CURR')
	appl_bureau_group = bureau.groupby('SK_ID_CURR')

	# active_bureau_loans = bureau.filter(lambd.groupby()
	num_active_credit = active_credit_group.apply(len)
	num_credit_bureau_loans = appl_bureau_group.apply(len)

	train = merge_grouping_in_sk_id(train, num_active_credit, 'num_active_cbs')
	test = merge_grouping_in_sk_id(test, num_active_credit, 'num_active_cbs')

	train_center_features = [pd.DataFrame(dist_from_center(train, feat)) for feat in CENTER_FEATURES]
	test_center_features = [pd.DataFrame(dist_from_center(test, feat)) for feat in CENTER_FEATURES]
	
	train = pd.concat([train] + train_center_features, axis=1)
	test = pd.concat([test] + test_center_features, axis=1)

	# contact features
	train['contact_features_sum'] = train[CONTACT_FEATURES].apply(sum)
	test['contact_features_sum'] = test[CONTACT_FEATURES].apply(sum)

	print('CURRENT FEATURES')
	for c in train.columns:
		print(c)

	return train, test




def categorical_conversion(data, test):
	
	categorical_feats = data.select_dtypes('O').columns
	for f_ in categorical_feats:
	    data[f_], indexer = pd.factorize(data[f_])
	    test[f_] = indexer.get_indexer(test[f_])
	return data, test


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





def train_model(data_, test_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]


    # params = { 

    # 		n_estimators:4000,
    #         learning_rate:0.03,
    #         num_leaves:30,
    #         colsample_bytree:.8,
    #         subsample:.9,
    #         max_depth:7,
    #         reg_alpha:.1,
    #         reg_lambda:.1,
    #         min_split_gain:.01,
    #         min_child_weight:2,
    #         silent:-1,
    #         verbose:-1,
    #         }
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=30,
            colsample_bytree=.8,
            subsample=.9,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
               )
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
    
    test_['TARGET'] = sub_preds

    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df
    




if __name__ == "__main__":

	data = pd.read_csv('../data/application_train.csv')
	test = pd.read_csv('../data/application_test.csv')

	# TESTING
	# data = data.__next__()
	# test = test.__next__()

	kf = KFold(n_splits=FOLDS)


	# bureau_bal = pd.read_csv('../data/bureau_balance.csv')
	# pos_cash_bal = pd.read_csv('../data/POS_CASH_balance.csv')
	# credit_card_bal = pd.read_csv('../data/credit_card_balance.csv')
	# prev_app = pd.read_csv('../data/previous_application.csv')
	# inst_payments = pd.read_csv('../data/installments_payments.csv')

	data, test = categorical_conversion(data, test)
	y = data.TARGET
	del data['TARGET']
	gc.collect()

	# data, test = get_features(data, test)
	oof_preds, submission, feature_importance = train_model(data, test, y, kf)
	print('DONE')

