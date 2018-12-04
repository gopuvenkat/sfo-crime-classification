import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import pickle

class Model:
	def __init__(self, cross_val_mode = False):
		self.train = pd.read_csv('./dataset/fe_train.csv', parse_dates=['Dates'])
		self.test = pd.read_csv('./dataset/fe_test.csv', parse_dates=['Dates'])
		self.cross_val_mode = cross_val_mode
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.sf = ['X', 'Y', 'Hour', 'Minutes', 'Year', 'Month', 'Day', 'DayOfWeekNum', 'PdDistrictNum', 'Address_CrossRoad', 'Address_clean_encode']
		self.fs1 = ['is_weekend', 'is_night_time', 'is_holiday']
		self.fs2 = ['AL', 'AR', 'AV', 'AY', 'BL', 'CR', 'CT', 'DR', 'ER', 'EX', 'HY', 'LN', 'MS', 'NO', 'PL', 'PZ', 'RD', 'RK', 'RW', 'ST', 'TI', 'TR', 'WK', 'WY']
		self.fs3 = ['is_avenue',  'is_street', 'is_block']
		self.fs4 = ['X_reduced', 'Y_reduced', 'rot_45_X', 'rot_45_Y', 'rot_30_X', 'rot_30_Y', 'rot_60_X', 'rot_60_Y', 'radial_r', 'XY_reduced']
		self.fs5 = ['newMin', 'seasons']
		self.features = self.sf + self.fs1 + self.fs3

	def split_data(self):
		if self.cross_val_mode:
			train_, test_ = train_test_split(
				self.train, 
				test_size = 0.3, 
				random_state = 3, 
				shuffle = True 
			)
			self.y_train = train_['CategoryNum']
			self.X_train = train_[self.features]
			self.y_test = test_['CategoryNum']
			self.X_test = test_[self.features]
		else:
			self.y_train = self.train['CategoryNum']
			self.X_train = self.train[self.features]
			self.X_test = self.test[self.features]

	def LogisticRegression(self):
		model = LogisticRegression(
			random_state=42, 
			solver='sag',
			multi_class='multinomial',
			max_iter=90)
		return model

	def NaiveBayes(self):
		model = GaussianNB()
		return model

	def DecisionTreeClassifier(self):
		max_depth = 8

		model = DecisionTreeClassifier(
			max_depth=max_depth)
		return model

	def RandomForestClassifier(self):
		random_state = 42
		max_depth = 17
		min_weight_fraction_leaf = 1
		n_estimators = 100
		n_jobs = -1

		model = RandomForestClassifier(
			random_state=random_state,
			max_depth=max_depth,
			n_estimators=n_estimators,
			n_jobs=n_jobs,
		#     min_weight_fraction_leaf=min_weight_fraction_leaf
		)
		return model

	def xgboost(self):
		seed = 42
		max_depth = 17
		learning_rate = 0.2
		min_child_weight = 1
		n_estimators = 100

		model = xgb.XGBClassifier(
			objective='multi:softprob', 
			seed=seed, 
			max_depth=max_depth,
			nthread=8,
			n_jobs=8,
		#     min_child_weight=min_child_weight,
		#     learning_rate=learning_rate,
			n_estimators = n_estimators
		)
		return model

	def lightgbm(self):
		train_data = lgb.Dataset(self.X_train, label=ytrain_)
		test_data = lgb.Dataset(self.X_test, label=ytest_)
		params = {
			'application' : 'multiclass',
			'is_unbalanced' : 'true',
			'boosting' : 'gbdt',
			'learning_rate' : 0.05,
			'num_class' : 36,
			'metric' : 'multi_logloss'
		}
		model = lgb.train(params, train_data)
		print("LogLoss score: ", log_loss(ytest_, model.predict(self.X_test)))
		return model

	def pca(self):
		pca = PCA(n_components=17)
		pca.fit(Xtrain_, ytrain_)
		Xtrain_pca = pca.transform(Xtrain_)
		Xtest_pca = pca.transform(self.X_test)
		score = -1 * cross_val_score(model, Xtrain_pca, ytrain_, scoring='neg_log_loss', cv=3, n_jobs=8)
		print("Score = {0:.6f}".format(score.mean()))
		print(score)

	def GridSearchCV(self, model, param_grid):
		model_gscv = GridSearchCV(
			estimator=model,
			scoring='neg_log_loss', 
			param_grid=param_grid, 
			cv = 3,
			n_jobs = -1
		)
		return model_gscv

	def calculate_cross_val(self, model):
		if(self.cross_val_mode):
			score = -1 * cross_val_score(
				model, 
				self.X_train, 
				self.y_train, 
				scoring='neg_log_loss', 
				cv=3, 
				n_jobs=8
			)
			print("Score = {0:.6f}".format(score.mean()))
			print(score)

	def fit(self, model):
		model.fit(self.X_train, self.y_train)

	def model_pickle(self, model, name):
		pickle.dump(model, open("./pickle/" + name + ".p", "wb"))

	def load_pickle(self, name):
		return pickle.load(open("./pickle/" + name + ".p", "rb"))

	def make_submission_file(self, model, pickle_name):
		predictions = model.predict_proba(self.X_test)
		submission = pd.DataFrame(predictions)
		submission.columns = sorted(self.train.Category.unique())
		submission['Id'] = test['Id']
		print(submission)
		submission.to_csv('submission_' + model + '.csv', index=False)
		self.model_pickle(model, pickle_name)

def main():
	cross_val_mode = True

	ml = Model(cross_val_mode)
	ml.split_data()
	df = ml.DecisionTreeClassifier()
	ml.model_pickle(df,'testing')


	# if cross_val_mode:
	# 	ml.calculate_cross_val(lr)
	# else:
	# 	ml.fit(lr)
	# 	ml.make_submission_file(lr, 'test')


if __name__ == '__main__':
	main()
