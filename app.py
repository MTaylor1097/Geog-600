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

Mx = SGM["Year"]
My = SGM['Age']

Fx = SGF["Year"]
Fy = SGF['Age']

plot1 = ax.scatter(Mx, My, label = 'Male Athletes')
plt.legend(loc = 'upper right', frameon=False)
ax.set_ylabel('Age')
ax.set_xlabel('Olympic Games')
ax.set_title('Ages of Gold Medalists')
plot2 =ax.scatter(Fx, Fy, label = 'Female Athletes')
plt.legend(loc = 'upper right', frameon=False)


              




    


                


