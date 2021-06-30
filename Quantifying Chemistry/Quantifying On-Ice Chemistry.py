#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Note: Decisions that I made throughout this process are explained in "Quantifying Chemsitry - Explanation on Decisions" doc
    #There is no reason to read through the whole document, but all decisions are in it if you are curious


# ## Importing the Dataset and Filtering Out Irrelevant Features

# In[2]:


#First let's import the original dataset with all of our player data info
import pandas as pd

chemistry_df = pd.read_csv('Chemistry_Original_df.csv')

chemistry_df.head()


# In[3]:


#Before doing anything, I want to add a new column which gives a unique code for every observation (player season)
#Players are in this dataset multiple times (if they played multiple seasons), so we want a unique code for each player season

Player_Season = chemistry_df['Player'] + chemistry_df['Season']
chemistry_df['Player_Season'] = Player_Season


# In[4]:


#Now we want to see all of our potential features, and filter out the ones that we don't want
#An explaation for why each feature was or was not chosen can be found under "Quantifying Chemsitry - Explanation on Decisions"

chemistry_df.columns


# In[5]:


chemistry_cluster_features = chemistry_df.drop(['Player', 'Season', 'Team', 'Position', 'GP', 'TOI','Points/60',
                                               'iSF/60', 'iFF/60', 'Sh%', 'FSh%', 'xFSh%', 'iPENT2/60', 'iPEND2/60',
                                                'iPENT5/60', 'iPEND5/60', 'iPEN±/60', 'FOW/60', 'FOL/60',
                                                'Player_Season', 'xGF%', 'xGF/60', 'xGA/60'], axis='columns')


# ## Preprocessing the Data

# In[6]:


#Before doing the clustering analysis, we want to scale our data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(chemistry_cluster_features)
StandardScaler(copy=True, with_mean=True, with_std=True)
chemistry_features_scaled = scaler.transform(chemistry_cluster_features)


# ## Building the Clustering Model

# In[7]:


#Before choosing how many clusters I want to use in the analysis, let's use the "elbow method" to chose a good amount

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

number_of_clusters = range(3, 9)
inertias = []

for k in number_of_clusters:
    model = KMeans(n_clusters=k)
    model.fit(chemistry_features_scaled)
    inertias.append(model.inertia_)
    
plt.plot(number_of_clusters, inertias, '-o')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.show()


# In[8]:


#We are going to go with 5 clusters, this decision is explained in the document
#Now we actually want to fit the model and assign each player to a cluster

model = KMeans(n_clusters = 5)
model.fit(chemistry_features_scaled)
cluster = model.predict(chemistry_features_scaled)


# ## Reconstructing the Original DF With New Player Clusters Added

# In[9]:


#Now I want to see which players are in which cluster, so first we will turn the predicted data into a df

Player_Clusters = pd.DataFrame(cluster)
Player_Clusters.head()


# In[10]:


#Now we want to add each players cluster back to the original dataframe with all of the player information

chemistry_df['Cluster'] = Player_Clusters

chemistry_df.head()


# ## Investigating Each Cluster

# In[11]:


#Now I want to get some more interpretable information on each cluster, and really understand why they are clustered this way
#First lets create new df for each cluster so that we can investigate them seperatly and in more depth

cluster_0 = chemistry_df[chemistry_df['Cluster']==0]
cluster_1 = chemistry_df[chemistry_df['Cluster']==1]
cluster_2 = chemistry_df[chemistry_df['Cluster']==2]
cluster_3 = chemistry_df[chemistry_df['Cluster']==3]
cluster_4 = chemistry_df[chemistry_df['Cluster']==4]


# In[35]:


#Let's start by investigating which players are in each cluster

cluster_0.Player_Season.unique()


# In[29]:


#Now lets see the averages of every category for each tier, which will help us get a better sense of the type of players in them
#This will also give us a baseline to work off of when we are looking at how well they play with each other

Cluster_0_mean = (cluster_0.mean())
Cluster_1_mean = (cluster_1.mean())
Cluster_2_mean = (cluster_2.mean())
Cluster_3_mean = (cluster_3.mean())
Cluster_4_mean = (cluster_4.mean())

#And then lets create a dataframe with the results so that we can easily compare each cluster

Cluster_means = ([Cluster_0_mean, Cluster_1_mean, Cluster_2_mean, Cluster_3_mean, Cluster_4_mean])
Cluster_means_df = pd.DataFrame(Cluster_means)


# ## Testing Cluster Chemistry - Player Pairs

# In[13]:


#Within the scope of this analysis, we will only test the chemistry between 2 players
    #Future analysis could look into the chemistry between 3 players (a full line)

Player_Pairs = pd.read_csv('Player_Pairs.csv')

Player_Pairs.head()


# In[14]:


Player_Pairs.shape


# In[15]:


#So we have 27,670 different player pairs to work off of!
#Before doing anything, we need to add the specific player codes (player+season) so that we can match the players from both df's

Player_Season = Player_Pairs['Player'] + Player_Pairs['Season']
Teammate_Season = Player_Pairs['Teammate'] + Player_Pairs['Season']

Player_Pairs['Player_Season'] = Player_Season
Player_Pairs['Teammate_Season'] = Teammate_Season

Player_Pairs.head()


# In[16]:


#Now we need to create a function which identifies the player codes in this new df and then adds the players cluster

#First, I am going to create a dictionary with the player code as the key and the players cluster as the value
chemistry_dictionary = chemistry_df.set_index('Player_Season').to_dict()['Cluster']

#Now we will define a function which looks for the players code and then, if found, gives us the players cluster
def match_PlayerCluster(player_name):
    for key, value in chemistry_dictionary.items():
        if player_name in key:
            return value

#Now we will use that function to find each player and teammates cluster, and add it to the df
Player_Pairs['Player_Cluster'] = Player_Pairs['Player_Season'].apply(match_PlayerCluster)
Player_Pairs['Teammate_Cluster'] = Player_Pairs['Teammate_Season'].apply(match_PlayerCluster)

Player_Pairs.head()


# In[17]:


#Let's now split the data into a df for each cluster
#Every pair of players is represented twice in the dataset, each player once as the "Player" and once as the "Teammate"
    #So we only need to care about when players show up in the "Player" column as not to double count

Cluster0_teammate = Player_Pairs[Player_Pairs['Player_Cluster']==0]
Cluster1_teammate = Player_Pairs[Player_Pairs['Player_Cluster']==1]
Cluster2_teammate = Player_Pairs[Player_Pairs['Player_Cluster']==2]
Cluster3_teammate = Player_Pairs[Player_Pairs['Player_Cluster']==3]
Cluster4_teammate = Player_Pairs[Player_Pairs['Player_Cluster']==4]


# In[21]:


#Now we will create a df for each cluster, showing how well they play with each other cluster

#Creating a df of the average performance of cluster 0 players with each type of cluster

#First I am creating different df's for each type of cluster that cluster 0 plays with
Cluster0_teammate0 = Cluster0_teammate[Cluster0_teammate['Teammate_Cluster']==0]
Cluster0_teammate1 = Cluster0_teammate[Cluster0_teammate['Teammate_Cluster']==1]
Cluster0_teammate2 = Cluster0_teammate[Cluster0_teammate['Teammate_Cluster']==2]
Cluster0_teammate3 = Cluster0_teammate[Cluster0_teammate['Teammate_Cluster']==3]
Cluster0_teammate4 = Cluster0_teammate[Cluster0_teammate['Teammate_Cluster']==4]

#Now I am getting the mean performance of each df
Cluster0_teammate0_mean = (Cluster0_teammate0.mean())
Cluster0_teammate1_mean = (Cluster0_teammate1.mean())
Cluster0_teammate2_mean = (Cluster0_teammate2.mean())
Cluster0_teammate3_mean = (Cluster0_teammate3.mean())
Cluster0_teammate4_mean = (Cluster0_teammate4.mean())

#Then taking those means and putting it in a df
Cluster0_teammate_means = ([Cluster0_teammate0_mean, Cluster0_teammate1_mean, Cluster0_teammate2_mean,
                           Cluster0_teammate3_mean, Cluster0_teammate4_mean])

Cluster0_teammate_means_df = pd.DataFrame(Cluster0_teammate_means)


# In[22]:


#Creating a df of the average performance of cluster 1 players with each type of cluster

Cluster1_teammate0 = Cluster1_teammate[Cluster1_teammate['Teammate_Cluster']==0]
Cluster1_teammate1 = Cluster1_teammate[Cluster1_teammate['Teammate_Cluster']==1]
Cluster1_teammate2 = Cluster1_teammate[Cluster1_teammate['Teammate_Cluster']==2]
Cluster1_teammate3 = Cluster1_teammate[Cluster1_teammate['Teammate_Cluster']==3]
Cluster1_teammate4 = Cluster1_teammate[Cluster1_teammate['Teammate_Cluster']==4]

Cluster1_teammate0_mean = (Cluster1_teammate0.mean())
Cluster1_teammate1_mean = (Cluster1_teammate1.mean())
Cluster1_teammate2_mean = (Cluster1_teammate2.mean())
Cluster1_teammate3_mean = (Cluster1_teammate3.mean())
Cluster1_teammate4_mean = (Cluster1_teammate4.mean())

Cluster1_teammate_means = ([Cluster1_teammate0_mean, Cluster1_teammate1_mean, Cluster1_teammate2_mean,
                           Cluster1_teammate3_mean, Cluster1_teammate4_mean])

Cluster1_teammate_means_df = pd.DataFrame(Cluster1_teammate_means)


# In[23]:


#Creating a df of the average performance of cluster 2 players with each type of cluster

Cluster2_teammate0 = Cluster2_teammate[Cluster2_teammate['Teammate_Cluster']==0]
Cluster2_teammate1 = Cluster2_teammate[Cluster2_teammate['Teammate_Cluster']==1]
Cluster2_teammate2 = Cluster2_teammate[Cluster2_teammate['Teammate_Cluster']==2]
Cluster2_teammate3 = Cluster2_teammate[Cluster2_teammate['Teammate_Cluster']==3]
Cluster2_teammate4 = Cluster2_teammate[Cluster2_teammate['Teammate_Cluster']==4]

Cluster2_teammate0_mean = (Cluster2_teammate0.mean())
Cluster2_teammate1_mean = (Cluster2_teammate1.mean())
Cluster2_teammate2_mean = (Cluster2_teammate2.mean())
Cluster2_teammate3_mean = (Cluster2_teammate3.mean())
Cluster2_teammate4_mean = (Cluster2_teammate4.mean())

Cluster2_teammate_means = ([Cluster2_teammate0_mean, Cluster2_teammate1_mean, Cluster2_teammate2_mean,
                           Cluster2_teammate3_mean, Cluster2_teammate4_mean])

Cluster2_teammate_means_df = pd.DataFrame(Cluster2_teammate_means)


# In[24]:


#Creating a df of the average performance of cluster 3 players with each type of cluster

Cluster3_teammate0 = Cluster3_teammate[Cluster3_teammate['Teammate_Cluster']==0]
Cluster3_teammate1 = Cluster3_teammate[Cluster3_teammate['Teammate_Cluster']==1]
Cluster3_teammate2 = Cluster3_teammate[Cluster3_teammate['Teammate_Cluster']==2]
Cluster3_teammate3 = Cluster3_teammate[Cluster3_teammate['Teammate_Cluster']==3]
Cluster3_teammate4 = Cluster3_teammate[Cluster3_teammate['Teammate_Cluster']==4]

Cluster3_teammate0_mean = (Cluster3_teammate0.mean())
Cluster3_teammate1_mean = (Cluster3_teammate1.mean())
Cluster3_teammate2_mean = (Cluster3_teammate2.mean())
Cluster3_teammate3_mean = (Cluster3_teammate3.mean())
Cluster3_teammate4_mean = (Cluster3_teammate4.mean())

Cluster3_teammate_means = ([Cluster3_teammate0_mean, Cluster3_teammate1_mean, Cluster3_teammate2_mean,
                           Cluster3_teammate3_mean, Cluster3_teammate4_mean])

Cluster3_teammate_means_df = pd.DataFrame(Cluster3_teammate_means)


# In[25]:


#Creating a df of the average performance of cluster 4 players with each type of cluster

Cluster4_teammate0 = Cluster4_teammate[Cluster4_teammate['Teammate_Cluster']==0]
Cluster4_teammate1 = Cluster4_teammate[Cluster4_teammate['Teammate_Cluster']==1]
Cluster4_teammate2 = Cluster4_teammate[Cluster4_teammate['Teammate_Cluster']==2]
Cluster4_teammate3 = Cluster4_teammate[Cluster4_teammate['Teammate_Cluster']==3]
Cluster4_teammate4 = Cluster4_teammate[Cluster4_teammate['Teammate_Cluster']==4]

Cluster4_teammate0_mean = (Cluster4_teammate0.mean())
Cluster4_teammate1_mean = (Cluster4_teammate1.mean())
Cluster4_teammate2_mean = (Cluster4_teammate2.mean())
Cluster4_teammate3_mean = (Cluster4_teammate3.mean())
Cluster4_teammate4_mean = (Cluster4_teammate4.mean())

Cluster4_teammate_means = ([Cluster4_teammate0_mean, Cluster4_teammate1_mean, Cluster4_teammate2_mean,
                           Cluster4_teammate3_mean, Cluster4_teammate4_mean])

Cluster4_teammate_means_df = pd.DataFrame(Cluster4_teammate_means)


# In[161]:


#Now we will test the chemistry between clusters by looking at each clusters performance over average with each other cluster
#I am just doing this by using the built in calculator to take the performance with each cluster minuse the average performance

#Starting with xGF%
xGFPercent_Over_Average_data = [['AllOff_NoDef', 0.8, -1.66, -1.66, -1.22, 1.08],
        ['Well_Rounded', 1.29, -0.71, -0.75, -1.31, 2.45],
        ['Opportunistic', 1.06, -0.98, -0.85, -0.83, 2.62],
        ['Grinders', 2.02, -1.02, -0.31, 0.35, 2.92],
        ['Shot_Generators', 0.44, -1.15, -0.75, -0.97, 1.39]]

xGFPercent_Over_Average = pd.DataFrame(xGFPercent_Over_Average_data,
                                      columns=['Cluster', 'xGF%_With_AllOff_NoDef','xGF%_With_Well_Rounded',
                                                'xGF%_With_Opportunistic', 'xGF%_With_Grinders', 'xGF%_With_Shot_Generators'])

xGFPercent_Over_Average


# In[160]:


#Now looking at xGF/60

xGFPer60_Over_Average_data = [['AllOff_NoDef', 0.18, -0.12, -0.18, -0.16, 0.13],
        ['Well_Rounded', 0.27, -0.04, -0.09, -0.18, 0.24],
        ['Opportunistic', 0.24, -0.06, -0.07, -0.17, 0.23],
        ['Grinders', 0.4, -0.01, -0.02, -0.08, 0.31],
        ['Shot_Generators', 0.21, -0.09, -0.12, -0.19, 0.16]]

xGFPer60_Over_Average = pd.DataFrame(xGFPer60_Over_Average_data,
                                      columns=['Cluster','xGF/60_With_AllOff_NoDef', 'xGF/60_With_Well_Rounded',
                                        'xGF/60_With_Opportunistic', 'xGF/60_With_Grinders', 'xGF/60_With_Shot_Generators',])

xGFPer60_Over_Average


# In[163]:


#Now looking at xGA/60

xGAPer60_Over_Average_data = [['AllOff_NoDef', 0.08, 0.05, -0.01, -0.03, 0.02],
        ['Well_Rounded', 0.16, 0.03, -0.02, -0.07, 0.02],
        ['Opportunistic', 0.15, 0.03, 0.01, -0.1, -0.01],
        ['Grinders', 0.24, 0.08, 0, -0.11, 0.06],
        ['Shot_Generators', 0.14, 0.03, -0.04, -0.09, 0.01]]

xGAPer60_Over_Average = pd.DataFrame(xGAPer60_Over_Average_data, columns=['Cluster',
                                            'xGA/60_With_AllOff_NoDef', 'xGA/60_With_Well_Rounded', 'xGA/60_With_Opportunistic',
                                            'xGA/60_With_Grinders', 'xGA/60_With_Shot_Generators'])

xGAPer60_Over_Average

