import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import seaborn as sns

class Visualization:
    def __init__(self):
        self.data = pd.read_csv('./dataset/fe_train.csv', parse_dates=['Dates'])

    def plot_crime_count_vs_months(self):
        pylab.rcParams['figure.figsize'] = (20.0, 20.0)
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
        ax1.plot(self.data.groupby('Month').size(), 'ro-')
        ax1.set_title ('All crimes')
        
        start = 1
        end = 12
        plt.hist(list(self.data.Month), 23)
        plt.ylabel('Total no. of crimes')
        plt.xlabel('Month of the year')
        ax1.xaxis.set_ticks(np.arange(start, end+1, 1))
        plt.show()

    def plot_crime_count_vs_hour(self):
        pylab.rcParams['figure.figsize'] = (20.0, 20.0)
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
        ax1.plot(self.data.groupby('Hour').size(), 'ro-')
        ax1.set_title ('All crimes')

        start = 0
        end = 23
        ax1.xaxis.set_ticks(np.arange(start, end+1, 1))
        p=list(self.data.Hour)
        plt.hist(p, 47)
        plt.ylabel('Total no. of crimes')
        plt.xlabel('Hour of the day')
        plt.show()
        
    def plot_crime_category_vs_month(self):
        crime_categories=list(self.data.Category.unique())
        p=[]
        for i in crime_categories:
            l=[0 for i in range(12)]
            for j in range(len(self.data.Month)):
                if(self.data.Category[j]==i):
                    l[self.data.Month[j]-1]+=1
            p.append(l)
        l=[0 for i in range(12)]
        names=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_nums = [i for i in range(12)]
        for i in range(len(p)):
            plt.bar(month_nums, p[i], bottom=l, edgecolor='white', width=1)
            for j in range(12):
                l[j]+=p[i][j]
        plt.xticks(month_nums, names)
        plt.legend(crime_categories, loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        plt.show()
        
    def plot_crime_count_vs_pdDistrict(self):
        crime_categories=list(self.data.Category.unique())
        p=[]
        dic ={'MISSION' : 1, 'SOUTHERN' : 2, 'BAYVIEW' : 3, 'CENTRAL' : 4, 'INGLESIDE' : 5, 'NORTHERN' : 6, 'RICHMOND' : 7, 'TARAVAL' : 8, 'TENDERLOIN' : 9, 'PARK' : 10}
        for i in crime_categories:
            l=[0 for i in range(10)]
            for j in range(len(self.data.PdDistrict)):
                if(self.data.Category[j]==i):
                    l[dic[self.data.PdDistrict[j]]-1]+=1
            p.append(l)
        
        for i in range(len(p)):
            x=0
            for j in range(len(p[i])):
                x+=p[i][j]
            for j in range(len(p[i])):
                p[i][j]=p[i][j]*100/x

        names=list(self.data.PdDistrict.unique())
        fig, ax = plt.subplots()
        im = ax.imshow(p, aspect=0.35)
        cbar = ax.figure.colorbar(im)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(crime_categories)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(crime_categories)
        
    def plot_crime_category_map(self):
        mapdata = np.loadtxt("./sf_map_copyright_openstreetmap_contributors.txt")

        lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
        clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
        datanew=self.data[self.data.Category=="ASSAULT"]
        datanew=datanew[datanew.Y<90]
        ax1.scatter(datanew.X,datanew.Y,s=0.2)
        ax1.get_xaxis().set_ticks([])
        ax1.get_xaxis().set_ticklabels([])
        ax1.get_yaxis().set_ticks([])
        ax1.get_yaxis().set_ticklabels([])
        plt.show()
        
    def plot_all_crimes_map(self):
        mapdata = np.loadtxt("./sf_map_copyright_openstreetmap_contributors.txt")

        lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
        clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
        datanew1=self.data[self.data.Y<90]
        ax1.scatter(datanew1.X,datanew1.Y,s=0.2)
        ax1.get_xaxis().set_ticks([])
        ax1.get_xaxis().set_ticklabels([])
        ax1.get_yaxis().set_ticks([])
        ax1.get_yaxis().set_ticklabels([])
        plt.show()
        
    def plot_crime_count_vs_street(self):
        l = self.data.Address_clean_encode.value_counts().sort_values(ascending = False)
        val = list(l)
        val = val[1:11]
        key = list(l.keys())
        key = key[1:11]
        keyname=[]
        for i in range(10):
            for j in range(len(self.data.Address_clean_encode)):
                if(self.data.Address_clean_encode[j] == key[i]):
                    keyname.append(self.data.Address_clean[j])
                    break
        fig, ax = plt.subplots(figsize=(25, 7))
        sns.barplot( x=keyname, y=val)
        
    def plot_crime_count_vs_crossRoads(self):
        crime_categories=list(self.data.Category.unique())
        y1=[0 for i in range(36)]
        n1=[0 for i in range(36)]
        for i in range(len(self.data)):
            for j in range(len(crime_categories)):
                if(self.data.Category[i]==crime_categories[j]):
                    if(self.data.Address_CrossRoad[i] == True):
                        y1[j]+=1
                    else:
                        n1[j]+=1
        y2=[]
        for i in range(36):
            y2.append([y1[i], n1[i]])
        category_nums = [i for i in range(36)]
        plt.figure(figsize=(20,10))
        plt.bar(category_nums, y1, edgecolor='white', width=1)
        plt.bar(category_nums, n1, bottom=y1, edgecolor='white', width=1)
        l=["Crossroad", "Not a crossroad"]
        plt.legend(l, loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        plt.xticks(category_nums, crime_categories, rotation=90)
        plt.show()
    
def main():
    v = Visualization()
    v.plot_crime_count_vs_months()
    v.plot_crime_count_vs_hour()
    v.plot_crime_category_vs_month()
    v.plot_crime_count_vs_pdDistrict()
    v.plot_crime_category_map()
    v.plot_crime_count_vs_street()
    v.plot_all_crimes_map()
    v.plot_crime_count_vs_crossRoads()

if __name__ == '__main__':
	main()
