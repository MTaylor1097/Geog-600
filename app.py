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
# plotF = plt.scatter(Fx, Fy, s= 8, label = "Female Athletes")
# plt.legend(loc = "upper right")
# plt.title("Age of Gold Medalists, 1896-2016")
# plt.xlabel("Olympic Year")
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

# #Male
# plt.plot(df_m.index, df_m['Height'], 'bo', markersize = 2, alpha = 0.3, label = 'Height (cm)')
# plt.xlabel('Olympic Year')
# plt.ylabel('Height(cm)')
# plt.legend(loc = 'upper center')
# plt.twinx()
# plt.plot(df_m.index+1, df_m['Weight'], 'ro', markersize = 2, alpha = 0.3, label = 'Weight (kg)')
# plt.ylabel('Weight (kg)')
# plt.legend()
# plt.title('Heigt and Weight comparison')
# plt.savefig('Dual_Axis_Male.png')

# #Female
# plt.plot(df_f.index, df_f['Height'], 'bo', markersize = 2, alpha = 0.3, label = 'Height (cm)')
# plt.xlabel('Olympic Year')
# plt.ylabel('Height(cm)')
# plt.legend(loc = 'upper center')
# plt.twinx()
# plt.plot(df_f.index+1, df_f['Weight'], 'ro', markersize = 2, alpha = 0.3, label = 'Weight (kg)')
# plt.ylabel('Weight (kg)')
# plt.legend()
# plt.title('Heigt and Weight comparison')
# plt.savefig('Dual_Axis_Female.png')

#Twin x axis plot of average heights and weights at each olympics
m_meanx = m_group['Year'].apply(np.mean)
f_meanx = f_group['Year'].apply(np.mean)

m_mean_H = m_group['Height'].apply(np.mean)
f_mean_H = f_group['Height'].apply(np.mean)

m_mean_W = m_group['Weight'].apply(np.mean)
f_mean_W = f_group['Weight'].apply(np.mean)

#Male
plt.plot(m_meanx, m_mean_H, '-', markersize = 4, alpha = 0.3, label = 'Height (cm)' )
plt.xlabel('Olympic Year')
plt.ylabel('Height(cm)')
plt.legend(loc = 'upper center')
plt.twinx()
plt.plot(m_meanx, m_mean_W, '-', markersize = 4, alpha = 0.3, label = 'Weight (cm)', color = 'red)
plt.xlabel('Olympic Year')
plt.ylabel('Weight(cm)')
plt.legend(loc = 'upper left')
plt.title('Male Mean Height and Weight comparison')
plt.savefig('Dual_Axis_Male_Mean.png')

#Female
plt.plot(f_meanx, f_mean_H, '-', markersize = 4, alpha = 0.3, label = 'Height (cm)')
plt.xlabel('Olympic Year')
plt.ylabel('Height(cm)')
plt.legend(loc = 'lower center')
plt.twinx()
plt.plot(f_meanx, f_mean_W, '-', markersize = 4, alpha = 0.3, label = 'Weight (cm)', color = 'red' )
plt.xlabel('Olympic Year')
plt.ylabel('Weight(cm)')
plt.legend(loc = 'lower right')
plt.title('Female Mean Height and Weight comparison')
plt.savefig('Dual_Axis_Female_Mean.png')
    


              




    


                


