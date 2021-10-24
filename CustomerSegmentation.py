#!/usr/bin/env python
# coding: utf-8

# ### Name: Nguyễn Thị Hồng Nhung
# ### Student ID: K184060744

# # Customer Segmentation Analysis 🕵️‍♀️

# ## Import libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import squarify 
import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Prepocessing

# In[3]:


data = pd.read_csv("Online-Retail-Data.csv")


# In[4]:


data.head(1)


# In[5]:


data.info()


# The data set has 541909 data objects and 8 attributes

# In[6]:


data.describe()


# In[7]:


#Remove data that don't needed for analysis
data.drop(['Description','StockCode'], axis = 1, inplace = True)
data.shape


# In[8]:


#Calculating the Missing Values % contribution in Data
total = data.isnull().sum()
percent = data.isnull().sum()/len(data)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# In[9]:


#Remove Missing Values
data.dropna(inplace = True)
data.shape


# In[10]:


#Remove canceled orders (in which Quantity is < 0)
data = data[data['Quantity']>0]
data = data[data['UnitPrice']>0]
data.shape


# In[11]:


# As customer clusters may vary by geography, we will restrict the data to only United Kingdom customers, which contains most of our customers historical data.
data = data[data.Country == 'United Kingdom']
data.shape


# In[12]:


data.describe()


# In[13]:


# Convert the type of InvoiceDate attribute from string to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# ## RFM Analysis

# ### Recency

# In[14]:


data.tail(2)


# In[15]:


# Compute the first and last date of transaction
min_date = data['InvoiceDate'].min()
max_date = data['InvoiceDate'].max()
print("First transaction date:", min_date)
print("Last transaction date:", max_date)


# #### The last transaction date is 09/12/2011 so we will use this day to calculate the Rencency

# In[16]:


# Calculate the delta time between the last transaction date and each transaction date
data['Recency'] = (max_date - data['InvoiceDate']).dt.days
data_r = data.groupby('CustomerID').agg({'Recency' : lambda x:x.min()}).reset_index()
data_r


# ### Frequency

# In[17]:


data_f = data.groupby('CustomerID').agg({'InvoiceNo' : lambda x:len(x)}).reset_index()
data_f.rename(columns = {'InvoiceNo':'Frequency'},inplace=True)
data_f.head()


# ### Monetary

# In[18]:


data['Monetary'] = data['Quantity']*data['UnitPrice']
data_m = data.groupby('CustomerID').agg({'Monetary' : lambda x:x.sum()}).reset_index()
data_m.head()


# In[19]:


# Merge the rfm data to get the final RFM dataset
rfm = pd.merge(data_r,pd.merge(data_f, data_m, on='CustomerID'), on='CustomerID')
rfm.head()


# In[20]:


rfm.describe()


# In[21]:


from scipy import stats

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}: Skew:{}".format(column,skew))
    return

plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
check_skew(rfm,'Recency')
plt.subplot(3, 1, 2)
check_skew(rfm,'Frequency')
plt.subplot(3, 1, 3)
check_skew(rfm,'Monetary')

plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)


# #### The data is highly skewed,therefore we will perform log transformations to reduce the skewness of each variable.I add a small constant as log transformation demands all the values to be positive.

# In[22]:


rfm_log = rfm.copy()
rfm_log = np.log(rfm_log+1)

plt.figure(figsize=(9, 9))

plt.subplot(3, 1, 1)
check_skew(rfm_log,'Recency')

plt.subplot(3, 1, 2)
check_skew(rfm_log,'Frequency')

plt.subplot(3, 1, 3)
check_skew(rfm_log,'Monetary')

plt.tight_layout()
plt.savefig('Transformed.png', format='png', dpi=1000)


# In[30]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=rfm_log.columns)


# In[ ]:





# ## Build K-means Model

# ### Finding the Optimal Number of Cluster by using Elbow Curve

# In[24]:


ssd = [] #ssd = sum of squared distances
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters=k,random_state=0).fit(rfm_scaled)
    ssd.append(kmeans.inertia_) 
  
plt.plot(K, ssd, 'bx-')
plt.xlabel('Values of k')
plt.ylabel('Sum of squared distances')
plt.title('The Elbow Method for optimal k')
plt.show()

plt.savefig('ElbowCurve.png', format='png', dpi=1000)


# We can see the drop in the sum of squared distance starts to slow down after k =4. Hence 4 is the optimal number of clusters for our analysis.

# In[25]:


# Fitting model with k=4
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(rfm_scaled)
kmeans.labels_

# assign the label
rfm['Cluster'] = kmeans.labels_+1
rfm.head()


# In[26]:


rfm_statistic = rfm.groupby(['Cluster']).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)
rfm_statistic
 


# ### Labelling Segmentation

# 1. Purchase long ago, purchased few and spent little -> Lost Cheap Customers
# 2. Bought most recently, most often and spend the most -> Best customers
# 3. Purchase long ago, but purchase not frequently and spend pretty much -> Almost lost
# 4. Haven't purchase for some time, but purchased frequently and spend a lot -> Loyal Customer

# In[27]:


rfm['Segment'] = ''
rfm.loc[(rfm['Cluster']== 1),'Segment'] = 'Lost Cheap Customers'
rfm.loc[(rfm['Cluster']== 2),'Segment'] = 'Best Customer'
rfm.loc[(rfm['Cluster']== 3),'Segment'] = 'Almost Lost'
rfm.loc[(rfm['Cluster']== 4),'Segment'] = 'Loyal Customer'
rfm


# In[28]:


rfm_statistic1 = rfm.groupby(['Segment']).agg({
        'CustomerID': ['count']
    }).round(0)
rfm_statistic1


# In[29]:


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
rfm.groupby('Segment').CustomerID.count().plot.pie(autopct = '%.2f%%', figsize = (6, 6), colors = colors)
plt.title('Percentage customer per segment',size = 14)
plt.tight_layout()
plt.savefig('PieChart.png', format='png', dpi=1000)


# 
