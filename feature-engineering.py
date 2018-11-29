import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    def __init__(self):
        self.data = pd.read_csv('./dataset/train.csv')
        self.test = pd.read_csv('./dataset/test.csv')

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

    def handle_season(month):
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
        self.data['seasons'] = data.Month.apply(lambda month : self.handle_season(month))
        self.test['seasons'] = test.Month.apply(lambda month : self.handle_season(month))

    def handle_PdDistrict(self):
        labelencoder = LabelEncoder()
        self.data['PdDistrictNum'] = labelencoder.fit_transform(data['PdDistrict'])
        self.test['PdDistrictNum'] = labelencoder.fit_transform(test['PdDistrict'])

    def handle_Category(self):
        labelencoder = LabelEncoder()
        self.data['CategoryNum'] = labelencoder.fit_transform(data['Category'])

    def handle_outliers(self):
        self.data = self.data[self.data.X < -121]
        self.data = self.data[self.data.Y < 40]
        self.test = self.test[self.test.X < -121]
        self.test = self.data[self.test.Y < 40]


if __name__ == '__main__':
    fe = FeatureEngineering()
