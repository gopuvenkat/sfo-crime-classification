import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import holidays

class FeatureEngineering:

	us_holidays = holidays.US()

	def __init__(self):
		self.data = pd.read_csv('./dataset/train.csv', parse_dates=['Dates'])
		self.test = pd.read_csv('./dataset/test.csv', parse_dates=['Dates'])
		self.all_tags = []

	def get_dataPoints(self):
		return (self.data, self.test)

	def handle_dates(self):
		data_week_dict = {
			'Monday': 1,
			'Tuesday':2,
			'Wednesday':3,
			'Thursday':4,
			'Friday':5,
			'Saturday':6,
			'Sunday':7
		}

		self.data['Hour'] = self.data.Dates.dt.hour
		self.data['Minutes'] = self.data.Dates.dt.minute
		self.data['Year'] = self.data.Dates.dt.year
		self.data['Month'] = self.data.Dates.dt.month
		self.data['Day'] = self.data.Dates.dt.day
		self.data['DayOfWeekNum'] = self.data['DayOfWeek'].replace(data_week_dict)

		self.test['Hour'] = self.test.Dates.dt.hour
		self.test['Minutes'] = self.test.Dates.dt.minute
		self.test['Year'] = self.test.Dates.dt.year
		self.test['Month'] = self.test.Dates.dt.month
		self.test['Day'] = self.test.Dates.dt.day
		self.test['DayOfWeekNum'] = self.test['DayOfWeek'].replace(data_week_dict)

	def handle_min(self, min):
		if(min < 15):
			return 0
		elif(min >= 15 and min < 30):
			return 15
		elif(min >= 30 and min < 35):
			return 30
		else:
			return 45

	def set_new_min(self):
		self.data['newMin'] = self.data.Minutes.apply(lambda min : self.handle_min(min))
		self.test['newMin'] = self.test.Minutes.apply(lambda min : self.handle_min(min))

	def handle_season(self, month):
		# https://www.studentflights.com.au/destinations/san-francisco/weather
		if month in [3, 4, 5]:
			return 1
		elif month in [6, 7, 8]:
			return 2
		elif month in [9, 10, 11]:
			return 3
		elif month in [12, 1, 2]:
			return 4

	def set_season(self):
		self.data['seasons'] = self.data.Month.apply(lambda month : self.handle_season(month))
		self.test['seasons'] = self.test.Month.apply(lambda month : self.handle_season(month))

	def handle_PdDistrict(self):
		labelencoder = LabelEncoder()
		self.data['PdDistrictNum'] = labelencoder.fit_transform(self.data['PdDistrict'])
		self.test['PdDistrictNum'] = labelencoder.fit_transform(self.test['PdDistrict'])

	def handle_Category(self):
		labelencoder = LabelEncoder()
		self.data['CategoryNum'] = labelencoder.fit_transform(self.data['Category'])

	def handle_outliers(self):
		self.data = self.data[self.data.X < -121]
		self.data = self.data[self.data.Y < 40]
		self.test = self.test[self.test.X < -121]
		self.test = self.test[self.test.Y < 40]

	def handle_addressCaps(self, i):
		s = ''
		for j in i.split():
			if(j.isupper()):
				s = s + ' ' + j
		return s[1:]

	def set_addressCaps(self):
		self.data['newAddress'] = self.data.Address.apply(lambda a: self.handle_addressCaps(a))
		self.test['newAddress'] = self.test.Address.apply(lambda a: self.handle_addressCaps(a))
		labelencoder = LabelEncoder()
		self.data['newAddressNum'] = labelencoder.fit_transform(self.data.newAddress)
		self.test['newAddressNum'] = labelencoder.fit_transform(self.test.newAddress)

	def handle_is_crossroads(self):
		self.data['Address_CrossRoad'] = self.data['Address'].str.contains('/')
		self.test['Address_CrossRoad'] = self.test['Address'].str.contains('/')

	def handle_address(self):
		topN_address_list = self.data['Address'].value_counts()
		topN_address_list = topN_address_list[topN_address_list >=100]
		topN_address_list = topN_address_list.index

		self.data['Address_clean'] = self.data['Address']
		self.test['Address_clean'] = self.test['Address']
		self.data.loc[~self.data['Address'].isin(topN_address_list), 'Address_clean'] = 'Others'
		self.test.loc[~self.test['Address'].isin(topN_address_list), 'Address_clean'] = 'Others'

		crossload = self.data[self.data['Address_clean'].str.contains('/')]
		crossroad_list = crossload['Address_clean'].unique()

		for address in crossroad_list:
			address_split = address.split('/')
			reverse_address = address_split[1].strip() + ' / ' + address_split[0].strip()
			self.data.loc[self.data['Address_clean'] == reverse_address, 'Address_clean'] = address
			self.test.loc[self.test['Address_clean'] == reverse_address, 'Address_clean'] = address
		crossload = self.data[self.data['Address_clean'].str.contains('/')]
		crossroad_list = crossload['Address_clean'].unique()

		labelencoder = LabelEncoder()
		self.data['Address_clean_encode'] = labelencoder.fit_transform(self.data['Address_clean'])
		self.test['Address_clean_encode'] = labelencoder.fit_transform(self.test['Address_clean'])

	def handle_weekends(self, day):
		if day in ['Friday', 'Saturday', 'Sunday']:
			return True
		else:
			return False

	def set_weekends(self):
		self.data['is_weekend'] = self.data.DayOfWeek.apply(lambda x : self.handle_weekends(x))
		self.test['is_weekend'] = self.test.DayOfWeek.apply(lambda x : self.handle_weekends(x))

	def handle_night_time(self, time):
			if time >= 22 or time <= 6:
				return True
			else:
				return False

	def set_night_time(self):
		self.data['is_night_time'] = self.data.Hour.apply(lambda x : self.handle_night_time(x))
		self.test['is_night_time'] = self.test.Hour.apply(lambda x : self.handle_night_time(x))

	def handle_holidays(self, date):
		if date in self.us_holidays:
			return True
		else:
			return False

	def set_holidays(self):
		self.data['is_holiday'] = self.data.Dates.dt.date.apply(lambda x: self.handle_holidays(x))
		self.test['is_holiday'] = self.test.Dates.dt.date.apply(lambda x: self.handle_holidays(x))

	def get_address_char(self, address):
		strings = address.strip().split('/')
		if(len(strings) == 1):
			return [strings[0].strip()[-2:].strip()]
		else:
			return [strings[0].strip()[-2:].strip(), strings[1][-2:].strip()]

	def get_all_tags(self, all_address):
		all_tags = []
		for address in all_address:
			tags = self.get_address_char(address)
			for tag in tags:
				if(len(tag) != 0 and tag.isdigit() == False):
					all_tags.append(tag)
		return list(set(all_tags))

	def set_all_tags(self):
		self.all_tags = self.get_all_tags(self.data.Address)

	def generate_dict(self, col):
		col = col[0]
		all_dict = {}
		for i in self.all_tags:
			all_dict[i]=0
		for i in col:
			all_dict[i]=1
		return all_dict

	def handle_tags(self):
		self.set_all_tags()
		self.data['tags'] = self.data.Address.apply(lambda x: self.get_address_char(x))
		self.test['tags'] = self.test.Address.apply(lambda x: self.get_address_char(x))
		all_dicts_data = self.data[['tags']].apply(self.generate_dict,axis=1)
		all_dicts_test = self.test[['tags']].apply(self.generate_dict,axis=1)

		data_dicts_pd = pd.DataFrame(list(all_dicts_data),index=self.data.index)
		test_dicts_pd = pd.DataFrame(list(all_dicts_test),index=self.test.index)

		self.data = pd.concat([self.data,data_dicts_pd],axis=1)
		self.test = pd.concat([self.test,test_dicts_pd],axis=1)

	def write_fe_csv(self):
		self.data.to_csv('fe_train.csv', index=False)
		self.test.to_csv('fe_test.csv', index=False)

def main(fe):
	fe.handle_dates()
	fe.set_new_min()
	fe.set_season()
	fe.handle_PdDistrict()
	fe.handle_Category()
	fe.handle_outliers()
	fe.set_addressCaps()
	fe.handle_is_crossroads()
	fe.handle_address()
	fe.set_weekends()
	fe.set_night_time()
	fe.set_holidays()
	fe.handle_tags()
	fe.write_fe_csv()
	# return fe.get_dataPoints()

if __name__ == '__main__':
	fe = FeatureEngineering()
	main(fe)
	data, test = fe.get_dataPoints()
	print(data.head())
