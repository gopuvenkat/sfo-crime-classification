#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss


# In[2]:


# Dates have to extracted
# data = pd.read_csv('train.csv', parse_dates=['Dates'])
data = pd.read_csv('./dataset/train.csv', parse_dates=['Dates'])
test = pd.read_csv('./dataset/test.csv', parse_dates=['Dates'])


# In[3]:


data.info()


# In[4]:


# No null values in the data-frame
data.isnull().values.any()


# In[5]:


# Dates
data_week_dict = {
    'Monday': 1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6,
    'Sunday':7
}

data['Hour'] = data.Dates.dt.hour
data['Minutes'] = data.Dates.dt.minute
data['Year'] = data.Dates.dt.year
data['Month'] = data.Dates.dt.month
data['Day'] = data.Dates.dt.day
data['DayOfWeekNum'] = data['DayOfWeek'].replace(data_week_dict)

test['Hour'] = test.Dates.dt.hour
test['Minutes'] = test.Dates.dt.minute
test['Year'] = test.Dates.dt.year
test['Month'] = test.Dates.dt.month
test['Day'] = test.Dates.dt.day
test['DayOfWeekNum'] = test['DayOfWeek'].replace(data_week_dict)


# In[6]:


data.head()


# In[7]:


def newMin(i):
    if(i<15):
        return 0
        
    elif(i>=15 and i<30):
        return 15

    elif(i>=30 and i<45):
        return 30
    
    elif(i>=45):
        return 45

data['newMin'] = data.Minutes.apply(lambda a:newMin(a))
test['newMin'] = test.Minutes.apply(lambda a:newMin(a))


# In[8]:


# https://www.studentflights.com.au/destinations/san-francisco/weather
def season(i):
    if i in [3, 4, 5]:
        return 1
    elif i in [6, 7, 8]:
        return 2
    elif i in [9, 10, 11]:
        return 3
    elif i in [12, 1, 2]:
        return 4
    
data['seasons'] = data.Month.apply(lambda a:season(a))
test['seasons'] = test.Month.apply(lambda a:season(a))


# In[9]:


labelencoder = LabelEncoder()


# In[10]:


data['ResolutionNum'] = labelencoder.fit_transform(data['Resolution'])
data['PdDistrictNum'] = labelencoder.fit_transform(data['PdDistrict'])
data['CategoryNum'] = labelencoder.fit_transform(data['Category'])

test['ResolutionNum'] = labelencoder.fit_transform(test['Resolution'])
test['PdDistrictNum'] = labelencoder.fit_transform(test['PdDistrict'])


# In[11]:


data = data[data.X < -121]
data = data[data.Y < 40]

test = test[test.X < -121]
test = test[test.Y < 40]


# In[12]:


def getCapsAddress(i):
    s=''
    for j in i.split():
        if(j.isupper()):
            s=s+' '+j
    return s[1:]

data['newAddress'] = data.Address.apply(lambda a:getCapsAddress(a))
test['newAddress'] = test.Address.apply(lambda a:getCapsAddress(a))


# In[13]:


data['newAddressNum'] = labelencoder.fit_transform(data.newAddress)
test['newAddressNum'] = labelencoder.fit_transform(test.newAddress)


# In[14]:


data['Address_CrossRoad'] = data['Address'].str.contains('/')
test['Address_CrossRoad'] = test['Address'].str.contains('/')

topN_address_list = data['Address'].value_counts()
topN_address_list = topN_address_list[topN_address_list >=100]
topN_address_list = topN_address_list.index
print(topN_address_list)

data['Address_clean'] = data['Address']
test['Address_clean'] = test['Address']
data.loc[~data['Address'].isin(topN_address_list), 'Address_clean'] = 'Others'
test.loc[~test['Address'].isin(topN_address_list), 'Address_clean'] = 'Others'
print(data.shape)

crossload = data[data['Address_clean'].str.contains('/')]
crossroad_list = crossload['Address_clean'].unique()
print(len(crossroad_list))


# In[15]:


for address in crossroad_list:
    address_split = address.split('/')
    reverse_address = address_split[1].strip() + ' / ' + address_split[0].strip()
    data.loc[data['Address_clean'] == reverse_address, 'Address_clean'] = address
    test.loc[test['Address_clean'] == reverse_address, 'Address_clean'] = address
crossload = data[data['Address_clean'].str.contains('/')]
crossroad_list = crossload['Address_clean'].unique()
print(len(crossroad_list))

le = LabelEncoder()
data['Address_clean_encode'] = le.fit_transform(data['Address_clean'])
print(data.shape)


# In[16]:


le = LabelEncoder()
test['Address_clean_encode'] = le.fit_transform(test['Address_clean'])


# In[17]:


def is_weekend(day):
    if day in ['Friday', 'Saturday', 'Sunday']:
        return True
    else:
        return False
    
data['is_weekend'] = data.DayOfWeek.apply(lambda x : is_weekend(x))
test['is_weekend'] = test.DayOfWeek.apply(lambda x : is_weekend(x))


# In[18]:


def night_time(time):
    if time >= 22 or time <= 6:
        return True
    else:
        return False

data['is_night_time'] = data.Hour.apply(lambda x : night_time(x))
test['is_night_time'] = test.Hour.apply(lambda x : night_time(x))


# In[19]:


import holidays
us_holidays = holidays.US()
def is_holiday(date):
    if date in us_holidays:
        return True
    else:
        return False

data['is_holiday'] = data.Dates.dt.date.apply(lambda x: is_holiday(x))
test['is_holiday'] = test.Dates.dt.date.apply(lambda x: is_holiday(x))


# In[20]:


def get_address_char(address):
    strings = address.strip().split('/')
    if(len(strings) == 1):
        return [strings[0].strip()[-2:].strip()]
    else:
        return [strings[0].strip()[-2:].strip(), strings[1][-2:].strip()]


# In[21]:


def get_tags(all_address):
    all_tags = []
    for address in all_address:
        tags = get_address_char(address)
        for tag in tags:
            if(len(tag) != 0 and tag.isdigit() == False):
                all_tags.append(tag)
    return list(set(all_tags))


# In[22]:


all_tags = get_tags(data.Address)


# In[23]:


data['tags'] = data.Address.apply(lambda x: get_address_char(x))
test['tags'] = test.Address.apply(lambda x: get_address_char(x))


# In[24]:


def makeDict(col):
    col = col[0]
    all_dict = {}
    for i in all_tags:
        all_dict[i]=0
    for i in col:
        all_dict[i]=1
    return all_dict


# In[25]:


all_dicts_data = data[['tags']].apply(makeDict,axis=1)
all_dicts_test = test[['tags']].apply(makeDict,axis=1)


# In[26]:


data_dicts_pd = pd.DataFrame(list(all_dicts_data),index=data.index)
test_dicts_pd = pd.DataFrame(list(all_dicts_test),index=test.index)


# In[27]:


# data_dicts_pd.drop(columns=['','80'],inplace=True)


# In[28]:


data = pd.concat([data,data_dicts_pd],axis=1)
test = pd.concat([test,test_dicts_pd],axis=1)


# In[35]:


corr = data.corr()
print(corr['CategoryNum'].sort_values(ascending=False))


# In[36]:


data["X_reduced"] = data.X.apply(lambda x: "{0:.2f}".format(x)).astype(float)
data["Y_reduced"] = data.Y.apply(lambda x: "{0:.2f}".format(x)).astype(float)
# data["X_reduced_cat"] = pd.Categorical.from_array(data.X_reduced).codes
# data["Y_reduced_cat"] = pd.Categorical.from_array(data.Y_reduced).codes

data["rot_45_X"] = .707*data["Y"] + .707*data["X"]
data["rot_45_Y"] = .707* data["Y"] - .707* data["X"]

data["rot_30_X"] = (1.732/2)*data["X"] + (1./2)*data["Y"]
data["rot_30_Y"] = (1.732/2)* data["Y"] - (1./2)* data["X"]

data["rot_60_X"] = (1./2)*data["X"] + (1.732/2)*data["Y"]
data["rot_60_Y"] = (1./2)* data["Y"] - (1.732/2)* data["X"]

data["radial_r"] = np.sqrt( np.power(data["Y"],2) + np.power(data["X"],2) )


# In[ ]:


data.loc[:,['X','Y','rot_45_X','rot_45_Y','rot_30_X','rot_30_Y','rot_60_X','rot_60_Y','radial_r']].head()


# In[37]:


features=['X','Y','Hour','Minutes','Year','Month','Day','DayOfWeekNum', 'PdDistrictNum',
          'Address_CrossRoad', 'Address_clean_encode','is_weekend', 'is_night_time', 'is_holiday'] + all_tags


# In[38]:


# for i in data.CategoryNum.unique():
#     print(i,labelencoder.inverse_transform(data.CategoryNum.unique())[i])
#     data[data.CategoryNum==i].hist(bins=50, figsize=(20,15))
#     plt.show()

# data.hist(bins=50, figsize=(20,15))
# plt.show()


# In[48]:


# Random seed has been set - As per the guidlines of the competition
train_, test_ = train_test_split(data, test_size=0.3, random_state=3, shuffle=True)


# In[49]:


ytrain_ = train_['CategoryNum']
Xtrain_ = train_[features]
ytest_ = test_['CategoryNum']
Xtest_ = test_[features]


# In[39]:


y_train = data['CategoryNum']
X_train = data[features]
# y_test = test['CategoryNum']
X_test = test[features]


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',max_iter=1000)
clf.fit(X_train,y_train)
pred = clf.predict_proba(X_test)
log_loss(y_test,pred)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(Xtrain_,ytrain_)
print(clf.score(Xtest_,ytest_))
pred = clf.predict_proba(Xtest_)
print(log_loss(ytest_,pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

max_depth = 8

model = DecisionTreeClassifier(
    max_depth=max_depth
)


# In[ ]:


importances = dt_model.feature_importances_
indices = np.argsort(importances)


# In[ ]:


plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[42]:


from sklearn.ensemble import RandomForestClassifier

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

# model.fit(Xtrain_,ytrain_)
# print(model.score(Xtest_,ytest_))
# pred = model.predict_proba(Xtest_)
# print(log_loss(ytest_,pred))


# In[50]:


import xgboost as xgb

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


# In[51]:


model


# In[52]:


score = -1 * cross_val_score(model, Xtrain_, ytrain_, scoring='neg_log_loss', cv=3, n_jobs=8)


# In[53]:


print("Score = {0:.6f}".format(score.mean()))
print(score)


# In[ ]:


# from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [17,19,20,21]
}
model_gscv = GridSearchCV(
    estimator=model,
    scoring='neg_log_loss', 
    param_grid=param_grid, 
    cv = 3,
    n_jobs = -1
)


# In[ ]:


model_gscv.fit(Xtrain_,ytrain_)


# In[ ]:


means = model_gscv.cv_results_['mean_test_score']
stds = model_gscv.cv_results_['std_test_score']
params = model_gscv.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


model.fit(X_train,y_train)


# In[ ]:


import pickle


# In[ ]:


pickle.dump(model, open("xgboost_wo_res61118.p", "wb"))


# In[ ]:


model = pickle.load(open("xgboost_wo_res61118.p", "rb"))
model


# In[45]:


predictions = model.predict_proba(X_test)


# In[46]:


submission = pd.DataFrame(predictions)
submission.columns = sorted(data.Category.unique())
submission['Id'] = test['Id']
submission


# In[47]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




