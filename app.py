#120 years of Olympic gold medal winners (summer games)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('athlete_events.csv')

fig, ax = plt.subplots()

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

m_meanx = m_group['Year'].apply(np.mean)
f_meanx = f_group['Year'].apply(np.mean)

#Plotting scatter

Mx = m_meanx
My = m_mean

Fx = f_meanx
Fy = f_mean

plotM = ax.scatter(Mx, My, s=8,label =  "Male Athletes")
plotF = ax.scatter(Fx, Fy, s= 8, label = "Female Athletes")
plt.legend(loc = "upper right")
ax.set_ylabel("Age")
ax.set_xlabel("Olympic Year")
ax.set_title("Age of Gold Medalists, 1896-2016")
plt.savefig('Scatter.png')

# # Plot distributions of male and female gold medalists
# # Need to remove missing data sets here as no NA values for Gold Medal winners age and when removed earlier removed values that can be plotted in scatter.

df = data.dropna()
summer_df = df[df['Season']=='Summer']
sum_gold_df = summer_df[summer_df["Medal"]== "Gold"]
SGM_df = sum_gold_df[sum_gold_df["Sex"]=="M"]
SGF_df = sum_gold_df[sum_gold_df["Sex"]=="F"]
labels = ['Male', 'Female']
width = 2


m_age = SGM_df["Age"]
f_age = SGF_df["Age"]


#plotting violin plot
plot_data = [m_age, f_age]
fig = plt.figure()
plt.savefig('Violin.png')
ax = fig.add_axes([1,1,1,1])

# #bp is the box plot
bp = ax.violinplot(plot_data, showmedians = True), (ax.set_ylabel('Age'), ax.set_title('Age Distribution of Gold Medal Winners'), ax.set_xticks([1,2]), ax.set_xticklabels(labels))
plt.show()
                         


              




    


                


