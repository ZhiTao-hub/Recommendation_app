#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Top 1000 Influencers include celebrities with large followings to niche content creators with a loyal following on YouTube, Instagram, Facebook, and Twitter.They are categorized in tiers (mini,mega), based on their number of followers.
# 
# Businesses pursue people who aim to lessen their consumption of advertisements, and are willing to pay their influencers more.
# 
# Market Researchers find that influencer selection extends into product personality. This product and benefit matching is key. For a shampoo, it should use an influencer with good hair.

# In[2]:


tiktok_path = 'C://Users/User/Documents/socialmedia/social media influencers - tiktok.csv'
instagram_path = 'C://Users/User/Documents/socialmedia/social media influencers - instagram.csv'
youtube_path = 'C://Users/User/Documents/socialmedia/social media influencers - youtube.csv'

df_t=pd.read_csv(tiktok_path)
df_i=pd.read_csv(instagram_path)
df_y=pd.read_csv(youtube_path)


# # tiktok data set analysis

# In[3]:


df_t.head(3)


# - In tiktok dataset there is not category and audience country data  so I didn't find it useful to tell a buisnessmen that which influencer must be approached by them categorywise

# ## Data Preprocessing 

# In[4]:


df_t.isnull().sum()


# In[5]:


df_t['Subscribers count'].str[-1].value_counts()


# In[6]:


import re
def convert(x):
    return re.findall('\d+\.?\d*',x)


# In[7]:


def change(df,list1):
    for i in list1:
        df['new'+i]=df[i].apply(convert)
        df['new'+i]=df['new'+i].apply(lambda x: "".join(x))
        df['new'+i]=pd.to_numeric(df['new'+i])
        df['new'+i]=np.where(['M' in j for j in df[i]],df['new'+i]*1000000,
                             np.where(['K' in j1 for j1 in df[i]],df['new'+i]*1000,df['new'+i]))
    return df
    


# In[8]:


change(df_t,['Subscribers count'])


# # TOP 10 most followed celebrity on Tiktok

# In[9]:


df_t.sort_values(by='newSubscribers count',ascending=False,ignore_index=True).iloc[0:10,[1,2]]


# # now analyse instagram and youtube data set

# In[10]:


df_i.head(2)
# instagram dataset


# In[11]:


df_y.head(2)
# youtube dataset


# ### Renaming some columns in instagram dataframe for convenience

# In[12]:


df_i.rename({'category_1':'Category','Audience country(mostly)':'Audience Country'},axis=1,inplace=True)

df_y.rename({'Subscribers':'Followers',},axis=1,inplace=True)


# In[13]:


df_i.head(2)


# In[14]:


df_y.head(2)


# In[15]:


df_i.isnull().sum()


# In[16]:


df_y.isnull().sum()


# In[17]:


df_i.drop_duplicates(subset=['Influencer insta name'],inplace=True)


# In[18]:


df_i.shape


# In[19]:


df_i.drop(labels=['Influencer insta name','Authentic engagement\r\n'],axis=1,inplace=True)


# In[20]:


df_i.head(2)


# In[21]:


li=['Followers','Engagement avg\r\n']


# In[22]:


change(df_i,li)


# ### Engagement rate : the Percentage of Followers who really engages with the content posted by Influencers 
# 
# #### Why ER  is so Important?
# 1. Good ER means your content is making an impact on audience(they really  like you)
# 2. The higher the engagement ,the more likely it is that the content will be boosted in the newsfeed and attracting more eyes.

# ##### Engagement Rate formula:
#     ER=(Engagement Average/total Followers)*100

# In[23]:


df_i['Engagement Rate']=np.round((df_i['newEngagement avg\r\n']/df_i['newFollowers'])*100,3)


# In[24]:


print(df_i['Followers'].str[-1].unique())


# In[25]:


# for convenice 
df_i['newFollowers']=df_i['newFollowers']/1000000


# In[26]:


df_i.drop(labels=['Engagement avg\r\n','newEngagement avg\r\n'],axis=1,inplace=True)


# In[27]:


df_i.head(5)


# ### TOP 15 most followed celebrity on  instagram

# In[28]:


df_i.sort_values(by='newFollowers',ascending=False,ignore_index=True).iloc[0:15,[0,1,3,-1]]


# In[29]:


plt.title('Top 15 most followed celebrity on instagram')
plt.xlabel('Followers in Million')
sns.barplot(y='instagram name',x='newFollowers',data=df_i.sort_values(by='newFollowers',ascending=False).head(15))


# In[30]:


pallete=['red','green','yellow','salmon','cyan','blue','orange']


# In[31]:


def plot(df):
    plt.figure(figsize=(8,6))
    plt.xlabel('number of times category occured')
    plt.ylabel('Category')
    df['Category'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)


# ### TOP  categories followed on instagram(POPULAR CATEGORIES  ON INSTAGRAM)

# In[32]:


plot(df_i)
    


# ## TOP  categories followed on YOUTUBE(POPULAR CATEGORIES  ON YOUTUBE)

# In[33]:


df_y.drop_duplicates(subset=['channel name'],inplace=True)


# In[34]:


plot(df_y)


# ### Conclusion:
# 1. Some categories are not on both plateforms 
# 2. Some categories are more popular on instagram than youtube and vice versa
# 3. Example-EDUCATION and Animation is more popular on YOUTUBE the INSTAGRAM

# # Decide That where you want to make ads

# In[35]:


def plot_c(df):
    plt.figure(figsize=(10,8))
    plt.xlabel('number of times category occured')
    df['Audience Country'].value_counts().sort_values().plot.barh(color=pallete)


# ## TOP consumer countries of the influencers content on INSTAGRAM

# In[36]:


plot_c(df_i)


# ## TOP consumer countries of the influencers content on YOUTUBE

# In[37]:


plot_c(df_y)


# ### (TARGET COUNTRY FOR BUISNESS)Checking the demand for categories by Country wise

# -For understanding that where is the demand of product

# In[38]:


def demand(data,category):
    return data[data['Category']==category]['Audience Country'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)
    


# In[39]:


demand(df_y,'Education')


# In[40]:


demand(df_i,'Lifestyle')


# ### TOP 15 most followed channels on  youtube

# In[41]:


df_y.iloc[0:10,[1,2,3]]
# youtube dataset is already sorted


# In[42]:


ly=['Followers','avg views', 'avg likes', 'avg comments']


# - if you want to go with mini followers for advertisement on instagram

# In[43]:


df_i['newFollowers'].describe()


# In[44]:


df_i['newFollowers'].quantile(0.94)


# - I am taking 60M as a threshold means for instagram celebrity havning above 60M followers are considerd to be mega celebrity

# In[45]:


df_i.head(2)


# - if you  want to make ads by mini influencers 

# In[46]:


def for_mini_followers_instagram(coun,cat):
    df1=df_i[df_i['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']<60]
    return df1_mini.sort_values(by='Engagement Rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[47]:


for_mini_followers_instagram('India','Music')


# - if you want to make ads by mega influencers

# In[48]:


def for_mega_followers_instagram(coun,cat):
    df1=df_i[df_i['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']>60]
    return df1_mini.sort_values(by='Engagement Rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[49]:


for_mega_followers_instagram('India','Music')


# In[50]:


for_mini_followers_instagram('India','Beauty')


# In[51]:


for_mini_followers_instagram('India','Shows')


# In[52]:


#category(df_i,'Sports with a ball')


# In[53]:


for_mini_followers_instagram('India','Sports with a ball')


# In[54]:


for_mega_followers_instagram('India','Sports with a ball')


# In[55]:


df_y.head(3)


# In[56]:


df_y.isnull().sum()


# #### Due to nan values we have to remove some data .but in reality you can collect this data easily

# In[57]:


df_y.dropna(axis=0,how='any',subset=['avg likes','avg comments']).isnull().sum()


# In[58]:


df_y.dropna(axis=0,how='any',subset=['avg likes','avg comments'],inplace=True)


# In[59]:


change(df_y,ly)


# In[60]:


df_y[df_y['Audience Country']=='Spain']['Category'].value_counts()


# In[61]:


df_y[df_y['Audience Country']=='Brazil'].groupby('Category').get_group('Animation')


# In[62]:


df_y['Engagement rate']=round(((df_y['newavg comments']+df_y['newavg likes']+df_y['newavg views'])/df_y['newFollowers'])*100,3)


# In[63]:


df_y.head(2)


# In[64]:


df_y.columns


# In[65]:


# for convenince
df_y['newFollowers']=df_y['newFollowers']/1000000


# In[66]:


df_y.drop(labels=['avg views', 'avg likes', 'avg comments','newavg views', 'newavg likes', 'newavg comments',
       ],axis=1,inplace=True)


# In[67]:


df_y['newFollowers'].describe()


# In[68]:


df_y['newFollowers'].quantile(0.90)


# ### Threshold can be decided by your choice 
# 
# - Here i am cosidering that who have >30M subscribers that is coming the category of mega celebrity

# In[69]:


df_y.head(1)


# In[70]:


def for_mini_followers_youtube(coun,cat):
    df1=df_y[df_y['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']<=30]
    return df1_mini.sort_values(by='Engagement rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[71]:


df_y.groupby('Category')['Audience Country'].first()


# In[72]:


for_mini_followers_youtube('United States','Music & Dance')


# In[73]:


def for_mega_followers_youtube(coun,cat):
    df1=df_y[df_y['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']>30]
    return df1_mini.sort_values(by='Engagement rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[74]:


for_mega_followers_instagram('Brazil','Sports with a ball')


# In[75]:


df_y['Category'].value_counts()


# In[76]:


for_mega_followers_youtube('India','Movies')


# ## Visualization

# ### Tiktok 2022
# 
# - Content creators in TikTok have various features in their interface, but for obvious reasons these features are not included in the dataset, as the dataset focuses primarily on the popularity of the creator rather than other variables.¶
# 

# In[77]:


df_tiktok = pd.read_csv("C://Users/User/Documents/socialmedia/social media influencers - tiktok.csv")
df_youtube = pd.read_csv("C://Users/User/Documents/socialmedia/social media influencers - youtube.csv")
df_instagram = pd.read_csv("C://Users/User/Documents/socialmedia/social media influencers - instagram.csv")

df_tiktok.head(4)


# In[78]:


df_tiktok.nunique()


# In[79]:


df_tiktok.shape


# In[80]:


df_tiktok.info()


# In[81]:


df_tiktok.describe()


# In[82]:


# Convert an object to an int and add new columns
def data_to_numeric(df, column_list):
    for column_name in column_list:
        new_column_name = 'new_' + column_name
        if new_column_name not in df.columns:

            if df[column_name].dtype == 'object':
                df[new_column_name] = df[column_name].apply(str)
                df[new_column_name] = df[new_column_name].apply(lambda x: x.replace('M', '') if 'M' in x else x)
                df[new_column_name] = df[new_column_name].apply(lambda x: x.replace('K', '') if 'K' in x else x)
                df[new_column_name] = pd.to_numeric(df[new_column_name], downcast='integer')
                df[new_column_name] = np.where(df[column_name].str.contains('M'), df[new_column_name] * 1000000, 
                                                 np.where(df[column_name].str.contains('K'), df[new_column_name] * 1000, df[new_column_name]))
                df[new_column_name] = df[new_column_name].astype(int)
    return df
df_tiktok = data_to_numeric(df_tiktok, ['Subscribers count', 'Views avg.', 'Likes avg', 'Comments avg.', 'Shares avg'])
# df_tiktok.drop('new_Subscribers count', axis=1, inplace=True)
df_tiktok.head()


# In[83]:


sns.distplot(df_tiktok["new_Subscribers count"])


# ### The most followed Top 15 Tikokers

# In[84]:


plt.figure(figsize=(8, 10))
graph = sns.barplot(y='Tiktoker name',x='new_Subscribers count',data=df_tiktok.sort_values(by='new_Subscribers count',ascending=False).head(15), hue= 'Subscribers count', palette="tab10", clip_on=False)
graph.set(xlabel="Subscribers count", ylabel="Tiktoker name")
plt.show()


# ### Top 15 most liked Tikokers¶
# 

# In[85]:


graph = sns.barplot(y='Tiktoker name',x='new_Likes avg',data=df_tiktok.sort_values(by='new_Likes avg',ascending=False).head(15))
graph.set(xlabel="Subscribers count", ylabel="Tiktoker name")
plt.show()


# ### Top 15 most viewed Tiktokers

# In[86]:


data=df_tiktok.sort_values(by='new_Views avg.',ascending=False).head(15)
graph = sns.barplot(y='Tiktoker name',x='new_Views avg.', data = data)
graph.set(xlabel="Subscribers count", ylabel="Tiktoker name")
plt.show()


# ### Top 15 most shared Tikokers¶
# 

# In[87]:


data = df_tiktok.sort_values(by='new_Shares avg',ascending=False).head(15)
graph = sns.barplot(y='Tiktoker name',x='new_Shares avg', data=data)
plt.title("Top 15 most shared Tiktokers")
plt.xlabel("Shares avg")
plt.show()


# ## Youtube 2022

# In[88]:


df_youtube.head(10)


# As in Tiktok, YouTubers have other data, some of which is private and others of which are not added because their focus is on data related to popularity.¶
# 

# In[89]:


df_youtube.info()


# In[90]:


df_youtube.isnull().sum()


# In[91]:


df_youtube.describe()


# In[92]:


def data_to_numeric(df, column_list):
    for column_name in column_list:
        new_column_name = 'new_' + column_name
        if new_column_name not in df.columns:
            if df[column_name].dtype == 'object':
                df[new_column_name] = df[column_name].str.replace('M', '').str.replace('K', '')
                df[new_column_name] = pd.to_numeric(df[new_column_name], errors='coerce')
                # Handle "Unknown " values by replacing them with NaN
                df[new_column_name] = df[new_column_name].where(df[new_column_name].notnull(), other=None)
    return df
               


# In[93]:


# Convert an object to an int and add new columns
df_youtube = data_to_numeric(df_youtube, ['Subscribers', 'avg views', 'avg likes', 'avg comments'])
df_youtube = df_youtube.fillna("Unknown ")
df_youtube.head()


# The dataset does not record the number of videos shared or downloaded.

# In[94]:



sns.distplot(df_youtube["new_Subscribers"])


# ### Top 15 most followers Youtubers

# In[95]:


data = df_youtube.sort_values(by='new_Subscribers',ascending=False).head(15)
plt.figure(figsize=(8, 10))
graph = sns.barplot(y='channel name',x='new_Subscribers',data= data, hue= 'Subscribers', palette="tab10", clip_on=False)
graph.set(xlabel="Subscribers count", title = "Top 15 most followers Youtubers")
plt.show()


# ### Top 15 most viewed Youtubers¶
# 

# In[96]:


data = df_youtube.sort_values(by='new_avg views',ascending=False).head(15)
plt.figure(figsize=(8, 10))
graph = sns.barplot(y='channel name', x='new_avg views', data=data, hue='avg views', palette="tab10")
graph.set(xlabel="avg views", title="Top 15 most viewed Youtubers")
plt.legend(title='Average views')
plt.show()


# In[97]:


data = df_youtube.sort_values(by='new_avg views',ascending=False).head(15)
graph = sns.barplot(y='channel name', x='new_avg views', data=data)
graph.set(xlabel="avg views", title="Top 15 most viewed Youtubers")
plt.show()


# ### Most viewed categories

# In[98]:


data = df_youtube.sort_values(by='new_avg views',ascending=False)
gráfico = sns.barplot(y= 'Category' , x = 'new_avg views', data=data, ci = None)
graph.set(xlabel="avg views", title="Top 20 most viewed categories")
plt.savefig("most_viewed_categories_Youtube")
plt.show()


# ### Categories with more subscribers
# 

# In[99]:


data = df_youtube.sort_values(by='new_Subscribers',ascending=False)
# plt.figure(figsize = (12, 10))
graph = sns.barplot(y='Category', x='new_Subscribers', data=data, ci = None)
graph.set(xlabel="Subscribers")
plt.show()


# ### Instagram 2022

# In[100]:


df_instagram.shape


# In[101]:


df_instagram.head(5)


# Instagram does not provide aggregate data on average "Likes", "Views" and "Comments". This is in contrast to the previous two data sets, which did provide this information. Instead, Instagram provides data on the number of followers and encapsulates the average number of likes, comments and views in a single column. Engagement avg\r\n: This is a avergage of likes, views and comments of publications.

# In[102]:


df_instagram.describe()


# In[103]:


df_instagram.isnull().sum()


# In[104]:


df_instagram.info()


# In[105]:


df_instagram = data_to_numeric(df_instagram, ['Followers', 'Authentic engagement\r\n', 'Engagement avg\r\n'])
df_instagram = df_instagram.fillna("Unknow")
df_instagram.head()


# In[106]:


sns.distplot(df_instagram["new_Followers"])


# ### Top 16 most followers Instagramers¶
# 

# In[107]:


data = df_instagram.sort_values(by='new_Followers',ascending=False).head(16)
graph = sns.barplot(y='instagram name', x='new_Followers', data=data)
graph.set(xlabel="Followers")
plt.show()


# In[108]:


data = df_instagram.sort_values(by='new_Followers',ascending=False)
plt.figure(figsize = (10, 8))
graph = sns.barplot(y='category_1', x='new_Followers', data=data, ci = None)
graph.set(xlabel="Followers", ylabel = 'Category', title = 'Categories with more followers')
plt.show()


# In[109]:


data = df_instagram.sort_values(by='new_Engagement avg\r\n',ascending=False)
plt.figure(figsize = (10, 8))
graph = sns.barplot(y='category_1', x='new_Engagement avg\r\n', data=data, ci = None)
graph.set(xlabel="Engagement avg", ylabel = 'Category', title = 'Categories with more EDR')
plt.show()


# ## Build a recommendation system

# In[110]:



# Check the structure of the datasets
print("TikTok Data:")
print(df_tiktok.head())
print("\nInstagram Data:")
print(df_instagram.head())
print("\nYouTube Data:")
print(df_youtube.head())


# #### tiktok

# In[111]:


df_tiktok.head()


# In[112]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Feature Engineering
def extract_features(data):
    # Extract relevant features
    features = data[['new_Subscribers count', 'new_Likes avg', 'new_Comments avg.', 'new_Shares avg', 'new_Views avg.']]
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features


# In[113]:


# Calculate Similarity
def calculate_similarity(data):
    # Extract features
    features = extract_features(data)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features)
    
    return similarity_matrix


# In[114]:


# Generate Recommendations
def generate_recommendations(similarity_matrix, data, user_id, top_n=5):
    # Get the similarity scores for the given user
    user_similarity = similarity_matrix[user_id]
    
    # Sort the influencers based on similarity scores (descending order)
    sorted_indices = user_similarity.argsort()[::-1]
    
    # Exclude the user itself and select top N influencers
    top_influencers = sorted_indices[1:top_n+1]
    
    # Get the details of the top influencers
    recommendations = data.iloc[top_influencers]
    
    return recommendations


# In[115]:


# Example: Generate recommendations for a user on TikTok
user_id = 0  # Assuming user with ID 0
tiktok_similarity = calculate_similarity(df_tiktok)
tiktok_recommendations = generate_recommendations(tiktok_similarity, df_tiktok, user_id)
print("Recommendations for user on TikTok:")
print(tiktok_recommendations)


# In[116]:


# Tiktok Similarity Matrix
print("TikTok Similarity Matrix:")
print(tiktok_similarity)


# #### Instagram

# In[117]:


df_instagram.head()


# In[118]:


# Feature Engineering
def extract_features(data):
    # Extract relevant features
    features = data[['new_Followers', 'new_Authentic engagement\r\n', 'new_Engagement avg\r\n']]
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features

# Calculate Similarity
def calculate_similarity(data):
    # Extract features
    features = extract_features(data)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features)
    
    return similarity_matrix

# Generate Recommendations
def generate_recommendations(similarity_matrix, data, user_id, top_n=5):
    # Get the similarity scores for the given user
    user_similarity = similarity_matrix[user_id]
    
    # Sort the influencers based on similarity scores (descending order)
    sorted_indices = user_similarity.argsort()[::-1]
    
    # Exclude the user itself and select top N influencers
    top_influencers = sorted_indices[1:top_n+1]
    
    # Get the details of the top influencers
    recommendations = data.iloc[top_influencers]
    
    return recommendations


# Example: Generate recommendations for a user on Instagram 
user_id = 0  # Assuming user with ID 0
instagram_similarity = calculate_similarity(df_instagram)
instagram_recommendations = generate_recommendations(instagram_similarity, df_instagram, user_id)
print("Recommendations for user on instagram:")
print(instagram_recommendations)


# In[119]:


# Instagram Similarity Matrix
print("\nInstagram Similarity Matrix:")
print(instagram_similarity)


# #### Youtube

# In[120]:


df_youtube.head(100)


# In[121]:


# Feature Engineering
def extract_features(data):
    # Extract relevant features
    features = data[['new_Subscribers', 'new_avg views']]
    

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features

# Calculate Similarity
def calculate_similarity(data):
    # Extract features
    features = extract_features(data)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features)
    
    return similarity_matrix

# Generate Recommendations
def generate_recommendations(similarity_matrix, data, user_id, top_n=5):
    # Get the similarity scores for the given user
    user_similarity = similarity_matrix[user_id]
    
    # Sort the influencers based on similarity scores (descending order)
    sorted_indices = user_similarity.argsort()[::-1]
    
    # Exclude the user itself and select top N influencers
    top_influencers = sorted_indices[1:top_n+1]
    
    # Get the details of the top influencers
    recommendations = data.iloc[top_influencers]
    
    return recommendations


# Example: Generate recommendations for a user on Youtube
user_id = 0  # Assuming user with ID 0
youtube_similarity = calculate_similarity(df_youtube)
youtube_recommendations = generate_recommendations(youtube_similarity, df_youtube, user_id)
print("Recommendations for user on youtube:")
print(youtube_recommendations)


# In[122]:


# YouTube Similarity Matrix
print("\nYouTube Similarity Matrix:")
print(youtube_similarity)


# ### Generate Top-N Recommendations

# In[123]:


# Function to generate top-N recommendations based on similarity matrix
def generate_top_n_recommendations(similarity_matrix, influencers, top_n=5):
    top_n_recommendations = {}
    for idx, influencer in enumerate(influencers):
        # Exclude the influencer itself from the recommendations
        similar_indices = np.argsort(similarity_matrix[idx])[::-1][1:top_n+1]
        similar_influencers = [influencers[similar_idx] for similar_idx in similar_indices]
        top_n_recommendations[influencer] = similar_influencers
    return top_n_recommendations


# In[124]:


# Generate top-N recommendations for TikTok influencers
tiktok_influencers = df_tiktok['new_Subscribers count'].tolist()
tiktok_top_n_recommendations = generate_top_n_recommendations(tiktok_similarity, tiktok_influencers)


# In[125]:


# Generate top-N recommendations for Instagram influencers
instagram_influencers = df_instagram['new_Followers'].tolist()
instagram_top_n_recommendations = generate_top_n_recommendations(instagram_similarity, instagram_influencers)


# In[126]:


# Generate top-N recommendations for YouTube influencers
youtube_influencers = df_youtube['new_Subscribers'].tolist()
youtube_top_n_recommendations = generate_top_n_recommendations(youtube_similarity, youtube_influencers)


# In[127]:


# Print top-N recommendations for each platform
print("Top-N Recommendations for TikTok Influencers:")
for influencer, recommendations in tiktok_top_n_recommendations.items():
    print(f"{influencer}: {recommendations}")


# In[128]:


print("\nTop-N Recommendations for Instagram Influencers:")
for influencer, recommendations in instagram_top_n_recommendations.items():
    print(f"{influencer}: {recommendations}")


# In[129]:


print("\nTop-N Recommendations for YouTube Influencers:")
for influencer, recommendations in youtube_top_n_recommendations.items():
    print(f"{influencer}: {recommendations}")


# In[130]:


# Function to generate top-N recommendations for each influencer
def generate_top_n_recommendations(similarity_matrix, n=5):
    top_n_recommendations = []
    num_influencers = similarity_matrix.shape[0]
    
    for influencer_id in range(num_influencers):
        # Get similarity scores for the influencer
        similarity_scores = similarity_matrix[influencer_id]
        
        # Sort similarity scores and get indices of top-N influencers (excluding the influencer itself)
        top_n_indices = np.argsort(similarity_scores)[::-1][1:n+1]
        
        top_n_recommendations.append(top_n_indices)
    
    return top_n_recommendations


# In[131]:


# Generate top-N recommendations for each platform
tiktok_top_n_recommendations = generate_top_n_recommendations(tiktok_similarity)
instagram_top_n_recommendations = generate_top_n_recommendations(instagram_similarity)
youtube_top_n_recommendations = generate_top_n_recommendations(youtube_similarity)


# In[132]:


# Example: Print top-5 recommendations for the first influencer in each platform
print("Top-5 Recommendations for TikTok Influencers:")
print(tiktok_top_n_recommendations[0])
print("\nTop-5 Recommendations for Instagram Influencers:")
print(instagram_top_n_recommendations[0])
print("\nTop-5 Recommendations for YouTube Influencers:")
print(youtube_top_n_recommendations[0])


# In[ ]:





# ### User-Item recommandation System

# In[133]:


# Define a user-item recommendation system class
class UserItemRecommendationSystem:
    def __init__(self, tiktok_similarity, instagram_similarity, youtube_similarity):
        self.tiktok_similarity = tiktok_similarity
        self.instagram_similarity = instagram_similarity
        self.youtube_similarity = youtube_similarity


# In[134]:


def recommend_influencers_for_user(user_preference_vector, similarity_matrix, n=5):
    # Sort user preference vector to get indices of influencers
    sorted_indices = np.argsort(user_preference_vector)[::-1]
    
    # Filter out influencers already interacted with by the user
    filtered_indices = [idx for idx in sorted_indices if user_preference_vector[idx] == 0]
    
    # Select top-N influencers from filtered indices
    top_n_influencers = filtered_indices[:n]
    
    # Get similarity scores for the selected influencers
    similarity_scores = similarity_matrix[top_n_influencers]
    
    # Sort the influencers based on similarity scores and get their indices
    top_n_indices = np.argsort(similarity_scores)[::-1]
    
    return top_n_indices


# In[135]:


# Example: Recommend influencers for a user based on user preferences
user_preference_vector = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1])  # Example user preference vector
tiktok_recommendations = recommend_influencers_for_user(user_preference_vector, tiktok_similarity)
instagram_recommendations = recommend_influencers_for_user(user_preference_vector, instagram_similarity)
youtube_recommendations = recommend_influencers_for_user(user_preference_vector, youtube_similarity)


# In[136]:


print("Recommendations for the user:")
print("TikTok:", tiktok_recommendations)
print("Instagram:", instagram_recommendations)
print("YouTube:", youtube_recommendations)


# In[137]:


from sklearn.cluster import KMeans


# In[138]:


# Function to perform cluster analysis
def perform_cluster_analysis(similarity_matrix, num_clusters):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    
    return cluster_labels


# In[139]:


# Perform cluster analysis for each platform
num_clusters = 3  # Example: number of clusters
tiktok_cluster_labels = perform_cluster_analysis(tiktok_similarity, num_clusters)
instagram_cluster_labels = perform_cluster_analysis(youtube_similarity, num_clusters)
youtube_cluster_labels = perform_cluster_analysis(instagram_similarity, num_clusters)

print("Cluster Labels for TikTok Influencers:", tiktok_cluster_labels)
print("Cluster Labels for Instagram Influencers:", instagram_cluster_labels)
print("Cluster Labels for YouTube Influencers:", youtube_cluster_labels)


# ### Visualize the Similarity Network 

# In[140]:


import networkx as nx
import matplotlib.pyplot as plt


# In[141]:


# Function to visualize the similarity network
def visualize_similarity_network(similarity_matrix, cluster_labels):
    # Create a graph
    G = nx.Graph()

    # Add nodes (influencers) to the graph
    num_influencers = similarity_matrix.shape[0]
    for influencer_id in range(num_influencers):
        cluster_label = cluster_labels[influencer_id]
        G.add_node(influencer_id, cluster=cluster_label)

    # Add edges (similarities) to the graph
    for i in range(num_influencers):
        for j in range(i+1, num_influencers):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0.5:  # Adjust the threshold as needed
                G.add_edge(i, j, weight=similarity_score)
                
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_color=[cluster_labels[node] for node in G.nodes], cmap=plt.cm.Set1)
    plt.show() 
        


# In[142]:


# Visualize the similarity network for each platform
print("Similarity Network Visualization for TikTok Influencers:")
visualize_similarity_network(tiktok_similarity, tiktok_cluster_labels)


# In[143]:


print("Similarity Network Visualization for Instagram Influencers:")
visualize_similarity_network(instagram_similarity, instagram_cluster_labels)


# In[144]:


print("Similarity Network Visualization for YouTube Influencers:")
visualize_similarity_network(youtube_similarity, youtube_cluster_labels)


# In[30]:



# ### Collaborative Filtering

# import numpy as np
# 
# # Function to predict ratings for a target user using collaborative filtering
# def collaborative_filtering(similarity_matrix, user_ratings, target_user, n=5):
#     # Find similar users based on user-user similarity matrix
#     similar_users = np.argsort(similarity_matrix[target_user])[::-1][1:]
#     
#     # Predict ratings for unseen items (influencers)
#     predicted_ratings = np.mean(user_ratings[similar_users], axis=0)
#     
#     # Recommend top-N influencers with highest predicted ratings
#     top_n_recommendations = np.argsort(predicted_ratings)[::-1][:n]
#     
#     return top_n_recommendations

# # Example: Collaborative filtering for TikTok influencers
# tiktok_similarity_matrix = np.array([[1, 0.5, 0.2],
#                                      [0.5, 1, 0.3],
#                                      [0.2, 0.3, 1]])  # Example similarity matrix
# tiktok_user_ratings = np.array([[5, 4, 0],
#                                  [0, 3, 4],
#                                  [4, 0, 5]])  # Example user-item interactions (ratings)
# target_user = 2  # Example target user
# tiktok_recommendations = collaborative_filtering(tiktok_similarity_matrix, tiktok_user_ratings, target_user)
# print("TikTok Recommendations for User", target_user, ":", tiktok_recommendations)
# 

# # Function to perform collaborative filtering for Instagram influencers
# def collaborative_filtering_instagram(similarity_matrix_instagram, user_ratings_instagram, target_user_instagram, n=5):
#     similar_users_instagram = np.argsort(similarity_matrix_instagram[target_user_instagram])[::-1][1:]
#     predicted_ratings_instagram = np.mean(user_ratings_instagram[similar_users_instagram], axis=0)
#     top_n_recommendations_instagram = np.argsort(predicted_ratings_instagram)[::-1][:n]
#     return top_n_recommendations_instagram

# # Example: Collaborative filtering for Instagram influencers
# instagram_similarity_matrix = np.array([[1, 0.5, 0.2],
#                                      [0.5, 1, 0.3],
#                                      [0.2, 0.3, 1]])  # Example similarity matrix for Instagram
# instagram_user_ratings = np.array([[5, 4, 0],
#                                    [0, 3, 4],
#                                    [4, 0, 5]])  # Example user-item interactions (ratings) for Instagram
# target_user_instagram = 1  # Example target user for Instagram
# instagram_recommendations = collaborative_filtering_instagram(instagram_similarity_matrix, instagram_user_ratings, target_user_instagram)
# print("Instagram Recommendations for User", target_user_instagram, ":", instagram_recommendations)
# 

# # Function to perform collaborative filtering for YouTube influencers
# def collaborative_filtering_youtube(similarity_matrix_youtube, user_ratings_youtube, target_user_youtube, n=5):
#     similar_users_youtube = np.argsort(similarity_matrix_youtube[target_user_youtube])[::-1][1:]
#     predicted_ratings_youtube = np.mean(user_ratings_youtube[similar_users_youtube], axis=0)
#     top_n_recommendations_youtube = np.argsort(predicted_ratings_youtube)[::-1][:n]
#     return top_n_recommendations_youtube

# # Example: Collaborative filtering for YouTube influencers
# youtube_similarity_matrix = np.array([[1, 0.3, 0.5],
#                                       [0.3, 1, 0.4],
#                                       [0.5, 0.4, 1]])  # Example similarity matrix for YouTube
# youtube_user_ratings = np.array([[5, 4, 0],
#                                  [0, 3, 4],
#                                  [4, 0, 5]])  # Example user-item interactions (ratings) for YouTube
# target_user_youtube = 0  # Example target user for YouTube
# youtube_recommendations = collaborative_filtering_youtube(youtube_similarity_matrix, youtube_user_ratings, target_user_youtube)
# print("YouTube Recommendations for User", target_user_youtube, ":", youtube_recommendations)

# ### Content-based filtering systems

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Function to recommend influencers based on content-based filtering
# def content_based_filtering(similarity_matrix, user_interactions, target_user, n=5):
#     similar_influencers = np.argsort(similarity_matrix[user_interactions[target_user]])[::-1][1:]
#     top_n_recommendations = similar_influencers[:n]
#     return top_n_recommendations
# 
# # Example: Content-based filtering for TikTok influencers
# # Assume you have similarity_matrix_tiktok and user_interactions_tiktok for TikTok influencers
# similarity_matrix_tiktok = np.array([[1, 0.4, 0.6],
#                                      [0.4, 1, 0.2],
#                                      [0.6, 0.2, 1]])  # Example similarity matrix for TikTok
# user_interactions_tiktok = np.array([[1, 0, 1],
#                                      [0, 1, 1],
#                                      [1, 1, 0]])  # Example user interactions data for TikTok
# target_user_tiktok = 0  # Example target user for TikTok
# tiktok_recommendations = content_based_filtering(similarity_matrix_tiktok, user_interactions_tiktok, target_user_tiktok)
# print("TikTok Recommendations for User", target_user_tiktok, ":", tiktok_recommendations)
# 
# # Example: Content-based filtering for Instagram influencers
# # Assume you have similarity_matrix_instagram and user_interactions_instagram for Instagram influencers
# similarity_matrix_instagram = np.array([[1, 0.3, 0.5],
#                                         [0.3, 1, 0.4],
#                                         [0.5, 0.4, 1]])  # Example similarity matrix for Instagram
# user_interactions_instagram = np.array([[1, 0, 1],
#                                         [0, 1, 1],
#                                         [1, 1, 0]])  # Example user interactions data for Instagram
# target_user_instagram = 0  # Example target user for Instagram
# instagram_recommendations = content_based_filtering(similarity_matrix_instagram, user_interactions_instagram, target_user_instagram)
# print("Instagram Recommendations for User", target_user_instagram, ":", instagram_recommendations)
# 
# # Example: Content-based filtering for YouTube influencers
# # Assume you have similarity_matrix_youtube and user_interactions_youtube for YouTube influencers
# similarity_matrix_youtube = np.array([[1, 0.5, 0.2],
#                                       [0.5, 1, 0.3],
#                                       [0.2, 0.3, 1]])  # Example similarity matrix for YouTube
# user_interactions_youtube = np.array([[1, 0, 1],
#                                       [0, 1, 1],
#                                       [1, 1, 0]])  # Example user interactions data for YouTube
# target_user_youtube = 0  # Example target user for YouTube
# youtube_recommendations = content_based_filtering(similarity_matrix_youtube, user_interactions_youtube, target_user_youtube)
# print("YouTube Recommendations for User", target_user_youtube, ":", youtube_recommendations)

# - define a function content_based_filtering to perform content-based filtering for a target user.
# - assume that have actual similarity matrices and user interactions data for TikTok, Instagram, and YouTube influencers.
# - Wprovide examples of content-based filtering for TikTok, Instagram, and YouTube influencers for target users.

# import numpy as np
# 
# # Define collaborative similarity matrices for TikTok, Instagram, and YouTube (dummy data)
# collaborative_similarity_tiktok = np.random.rand(10, 10)  # Example collaborative similarity matrix for TikTok
# collaborative_similarity_instagram = np.random.rand(10, 10)  # Example collaborative similarity matrix for Instagram
# collaborative_similarity_youtube = np.random.rand(10, 10)  # Example collaborative similarity matrix for YouTube
# 
# # Define content-based similarity matrices for TikTok, Instagram, and YouTube (dummy data)
# content_similarity_tiktok = np.random.rand(10, 10)  # Example content-based similarity matrix for TikTok
# content_similarity_instagram = np.random.rand(10, 10)  # Example content-based similarity matrix for Instagram
# content_similarity_youtube = np.random.rand(10, 10)  # Example content-based similarity matrix for YouTube
# 
# def hybrid_recommendation(collaborative_similarity, content_similarity, target_user, n=5):
#     # Combine similarity matrices (you can use simple averaging here)
#     combined_similarity = (collaborative_similarity + content_similarity) / 2
#     
#     # Perform hybrid recommendation (example: recommend top n influencers)
#     # Your recommendation logic here
#     
#     return recommendations
# 
# # Example: Perform hybrid recommendation for TikTok
# target_user = 0  # Assuming user with ID 0
# tiktok_recommendations = hybrid_recommendation(collaborative_similarity_tiktok, content_similarity_tiktok, target_user)
# instagram_recommendations = hybrid_recommendation(collaborative_similarity_instagram, content_similarity_instagram, target_user)
# youtube_recommendations = hybrid_recommendation(collaborative_similarity_youtube, content_similarity_youtube, target_user)
# print("TikTok Recommendations for User", target_user, ":", tiktok_recommendations)
# print("Instagram Recommendations for User", target_user, ":", instagram_recommendations)
# print("YouTube Recommendations for User", target_user, ":", youtube_recommendations)

# import numpy as np
# 
# # Function to recommend influencers based on hybrid recommendation system
# def hybrid_recommendation(collaborative_similarity, content_similarity, target_user, n=5):
#     # Combine similarity matrices (you can use simple averaging here)
#     combined_similarity = (collaborative_similarity + content_similarity) / 2
#     
#     # Perform hybrid recommendation
#     similar_users = np.argsort(combined_similarity[target_user])[::-1][1:]
#     top_n_recommendations = similar_users[:n]
#     
#     return top_n_recommendations
# 
# # Example: Hybrid recommendation for TikTok, Instagram, and YouTube influencers
# # Assume you have actual similarity matrices for each platform
# # Replace the ellipsis (...) with actual similarity matrices for TikTok, Instagram, and YouTube
# collaborative_similarity_tiktok = np.array([[1, 0.4, 0.6],
#                                             [0.4, 1, 0.2],
#                                             [0.6, 0.2, 1]])  # Example collaborative similarity matrix for TikTok
# content_similarity_tiktok = np.array([[0.8, 0.3, 0.5],
#                                        [0.3, 0.9, 0.4],
#                                        [0.5, 0.4, 0.7]])  # Example content-based similarity matrix for TikTok
# 
# collaborative_similarity_instagram = np.array([[1, 0.3, 0.5],
#                                                 [0.3, 1, 0.4],
#                                                 [0.5, 0.4, 1]])  # Example collaborative similarity matrix for Instagram
# content_similarity_instagram = np.array([[0.7, 0.4, 0.6],
#                                          [0.4, 0.8, 0.2],
#                                          [0.6, 0.2, 0.9]])  # Example content-based similarity matrix for Instagram
# 
# collaborative_similarity_youtube = np.array([[1, 0.5, 0.2],
#                                              [0.5, 1, 0.3],
#                                              [0.2, 0.3, 1]])  # Example collaborative similarity matrix for YouTube
# content_similarity_youtube = np.array([[0.9, 0.2, 0.4],
#                                         [0.2, 0.7, 0.5],
#                                         [0.4, 0.5, 0.8]])  # Example content-based similarity matrix for YouTube
# 
# # Example target user
# target_user = 0
# 
# # Perform hybrid recommendation for TikTok
# tiktok_recommendations = hybrid_recommendation(collaborative_similarity_tiktok, content_similarity_tiktok, target_user)
# print("TikTok Recommendations for User", target_user, ":", tiktok_recommendations)
# 
# # Perform hybrid recommendation for Instagram
# instagram_recommendations = hybrid_recommendation(collaborative_similarity_instagram, content_similarity_instagram, target_user)
# print("Instagram Recommendations for User", target_user, ":", instagram_recommendations)
# 
# # Perform hybrid recommendation for YouTube
# youtube_recommendations = hybrid_recommendation(collaborative_similarity_youtube, content_similarity_youtube, target_user)
# print("YouTube Recommendations for User", target_user, ":", youtube_recommendations)
# 

# In[ ]:




