import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

data = pd.read_csv('athlete_events.csv')


#subset of summer games
summer = data[data['Season']=='Summer']

#subset of summer games gold medals
summer_gold = summer[summer["Medal"]== "Gold"]

#subsets of male and female summer gold medalists
SGM = summer_gold[summer_gold["Sex"]=="M"]
SGF = summer_gold[summer_gold["Sex"]=="F"]

#grouping to get mean ages
m_group = SGM.groupby(["Year"])
f_group = SGF.groupby(["Year"])

m_mean = m_group['Age'].apply(np.mean)
f_mean = f_group['Age'].apply(np.mean)

#mean of each year group so that x and y are the same size?
m_meanx = m_group['Year'].apply(np.mean)
f_meanx = f_group['Year'].apply(np.mean)

# #Scatter

# Mx = m_meanx
# My = m_mean

# Fx = f_meanx
# Fy = f_mean

# plotM = plt.scatter(Mx, My, s=8,label =  "Male Athletes")
# plotF = plt.scatter(Fx, Fy, s= 8, label = "Female Athletes", color = 'red')
# plt.legend(loc = "upper right")
# plt.title("Mean Age of Gold Medalists, 1896-2016")
# plt.xlabel("Olympic Year")
# x_ticks = np.arange(1896, 2024, 8)
# plt.xticks(x_ticks, rotation = '45')
# plt.ylabel("Age")
# plt.savefig('Scatter.png')
   

# #Histograms

# plt.hist(SGM['Age'], bins = 50, color = 'blue', label = 'Male', histtype = 'stepfilled', alpha = 0.3)
# plt.hist(SGF['Age'], bins = 50, color = 'red', label = 'Female', histtype = 'stepfilled', alpha = 0.3);
# plt.legend(loc = 'upper right')
# plt.title("Gold Medal Age Distributions")
# plt.xlabel("Age")
# plt.ylabel("Frequency Density")
# plt.savefig('Histograms.png')


# Time Series with two y axis

df_m = SGM.set_index('Year')
df_f = SGF.set_index('Year')

#Twin x axis plot of average heights and weights at each olympics
m_meanx = m_group['Year'].apply(np.mean)
f_meanx = f_group['Year'].apply(np.mean)

m_mean_H = m_group['Height'].apply(np.mean)
f_mean_H = f_group['Height'].apply(np.mean)

m_mean_W = m_group['Weight'].apply(np.mean)
f_mean_W = f_group['Weight'].apply(np.mean)

# #Male
# plt.plot(df_m.index, df_m['Height'], 'bo', markersize = 2, alpha = 0.2, label = 'Height (cm)')
# plt.plot(m_meanx, m_mean_H, '-', markersize = 10, label = 'Mean Height (cm)' )
# plt.xlabel('Olympic Year')
# x_ticks = np.arange(1896, 2024, 8)
# plt.xticks(x_ticks, rotation = '45')
# plt.ylabel('Height(cm)')
# plt.ylim(top = 240)
# plt.legend(bbox_to_anchor = (0.4, 0.8), frameon = False)
# plt.twinx()
# plt.plot(df_m.index+1.5, df_m['Weight'], 'ro', markersize = 2, alpha = 0.2, label = 'Weight (kg)')
# plt.plot(m_meanx, m_mean_W, '-', markersize = 10, label = 'Mean Weight (kg)', color = 'red')
# plt.ylabel('Weight (kg)')
# plt.ylim(top = 180)
# plt.legend(bbox_to_anchor = (0.4,0.65), frameon = False)
# plt.title('Male Height and Weight Comparison')
# plt.savefig('Dual_Axis_Male.png')

#Female
plt.plot(df_f.index, df_f['Height'], 'bo', markersize = 2, alpha = 0.2, label = 'Height (cm)')
plt.plot(f_meanx, f_mean_H, '-', markersize = 10, label = 'Mean Height (cm)')
plt.xlabel('Olympic Year')
x_ticks = np.arange(1900, 2024, 8)
plt.xticks(x_ticks, rotation = '45')
plt.ylabel('Height(cm)')
plt.ylim(top = 220)
plt.legend(bbox_to_anchor = (0.40, 1), frameon = False)
plt.twinx()
plt.plot(df_f.index+1.5, df_f['Weight'], 'ro', markersize = 2, alpha = 0.2, label = 'Weight (kg)')
plt.plot(f_meanx, f_mean_W, '-', markersize = 10, label = 'Mean Weight (kg)', color = 'red')
plt.ylabel('Weight (kg)')
plt.ylim(top = 160)
plt.legend(bbox_to_anchor = (0.40, 0.85), frameon = False)
plt.title('Female Height and Weight Comparison')
plt.savefig('Dual_Axis_Female.png')

# #Groupby Table

# group = summer_gold.groupby(['Year', 'Sex'])
# table1 = group.agg({'Age': 'mean', 'Weight' : 'mean', 'Height' : 'mean'})
# print(table1)



              




    


                


