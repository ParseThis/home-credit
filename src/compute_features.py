""" compute features """


# distance from mean request amount, sq, hamming, euclidean, absoulte
# distance from mean number of credit lines extended * 
# number of credit lines extended *
# has defaulted in  thepast *
# type of loan [ base feature ]
# complete load application 
# mean of categorical features.
# age group features
# distance from age group, 
# distance from mean amoutnt credit
# distance amount goods prices. 
# if distance given contract type [ amount goods price, amount, amoutn]
# lDAYS_LAST_PHONE_CHANGE

# get a sense of much the application deviates from the mean application.
# what is the mean application? 
# bucket days employed in > 0 ( is this an artifact of the data)


# phone based features
# distance from mean
# DAYS_LAST_PHONE_CHANGE


# DAYS_EMPLOYED & NAME_HOUSING_TYPE

# want to compute the distance from what is 
# expected. at a certain level. 

# what is the amount loaned for someone
# on a critical feature. Example. 

# how far greater, from the normal amount 
# loaned is the person who is unemployed	
# 	if less shoud this feature be zero


# how far from the normal goods price 
# is this is application 

# did this application just get a job 
# if so how much is the loan for?


# a decision tree should figure this out!

from sklearn.pipeline import Pipeline
from sklearn.base import 


EXCEPT_INT_FEATURES = [ 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH' ]
EXT_FEATURES = ['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']


CONTACT_FEATURES = """ FLAG_MOBIL
	FLAG_EMP_PHONE  
	FLAG_WORK_PHONE 
	FLAG_CONT_MOBILE
	FLAG_PHONE      
	FLAG_EMAIL """.split('\n')

CONTACT_FEATURES = list(map(str.strip, 	CONTACT_FEATURES))

DAYS_FEATURES = """ 
		DAYS_BIRTH
		DAYS_EMPLOYED
		DAYS_REGISTRATION
		DAYS_ID_PUBLISH
		DAYS_LAST_PHONE_CHANGE
		DAYS_CREDIT
		DAYS_CREDIT_ENDDATE
		DAYS_ENDDATE_FACT
		DAYS_CREDIT_UPDATE
		DAYS_DECISION
		DAYS_FIRST_DRAWING
		DAYS_FIRST_DUE
		DAYS_LAST_DUE_1ST_VERSION
		DAYS_LAST_DUE
		DAYS_TERMINATION
		DAYS_INSTALMENT
		DAYS_ENTRY_PAYMENT""".split('\n')


DAYS_FEATURES = list(map(str.strip, DAYS_FEATURES))

def get_mean_categorical(feature):
	return feature.map((feature.value_counts() / len(sum_flag_documents)).to_dict())

def get_flag_document_features_mean(data):

	return data.filter(regex='FLAG_DOC').sum(axis=1)

def get_phone_features(data):
	return data.filter(regex='FLAG.*(MOBIL)|(PHONE)');


def apply_aggregate_function_to_instance(data, feature, func=None):
    if not func:
        # apply mean
        mapper = (data[feature].value_counts()/ len(data)).to_dict()
        return data[feature].map(mapper)
    else:
        raise NotImplementedError


def distance_features(data, feature, func=None):
	return 


def handle_outlier_days_emp():

	mean_days_emp = data[data.DAYS_EMPLOYED < 365243]['DAYS_EMPLOYED'].mean()
	data.DAYS_EMPLOYED.apply(lambda x: mean_days_emp if x >= 365243 else x)
# compute distance from 


def load_data():
	pass

def compute(train, test):
		
	# this is probably not that useful.
	# what I want to get a sense of is 
	# is the application amount outside of centrality?

	mean_amount_credit = data.AMT_CREDIT.mean()
	data['dist_from_mean_amt_credit'] = data.apply(lambda x: x.AMT_CREDIT - mean_amout_credit)
	FEATURES = [ x for x in data.columns if x!= 'SK_ID_CURR'] 

	# building features
	buiding_data_mask = train[BUILDING_FEATURES].isnull().apply(sum, axis=1)

	# complete building information
	data['complete_building_info'] = building_data_mask > 0
	data['sum_ext_features'] =  data[EXT_FEATURES].sum(axis=1)


	# compute mean_featuers on categorical variables


	# mean categorical features
	train[FEATURES].select_dtype(('int64', 'object'))
	return train, test

if __name__ == "__main__":
	
	compute()