
#SALARY

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Import the dataset
df = pd.read_csv("D:/DATASCIENCE/MachineLearningWorks/Salary Data.csv")
print(df)

#Length of the dataset
print(len(df))

#Display first 5 entries
print(df.head())

#Display last 5 entries
print(df.tail())

#Number of rows and columns in the dataset
print(df.shape)

#Describe the data
print(df.describe())

#Type of the dataset
print(type(df))

#Type of all columns in the data
print(df.dtypes)

#Number of rows for each columns
print(df.count())

#Checking for null values
print(df.isna())

#Number of null values in each columns
print(df.isna().sum())

#Drop the rows containing null values
print(df.dropna(inplace=True))

#Checking for duplicate values
print(df.duplicated())

#Sum of duplicate values
print(df.duplicated().sum())

#To find unique values
print(df.nunique())

#Information about the data
print(df.info())

#Extracting Independent Variable
x =df.iloc[:, [4,5]].values
print(x)

#Shape of x
print("x shape:",x.shape)

#Finding optimal number of clusters using the elbow method
#Initializing the list for the values of WCSS
wcss_list= []

#Using for loop for iterations from 1 to 10.
for i in range(1, 11):
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
 kmeans.fit(x)
 wcss_list.append(kmeans.inertia_)
print(wcss_list)

#Plot the graph
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

#Training the K-means model on a dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)
y_kmeans= kmeans.fit_predict(x)
print(y_kmeans)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(x)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(df['Years of Experience'], df['Salary'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('K-Means Clustering')
plt.show()

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Print the centroids
print("Centroids:", centroids)

# Display the clustered data
print(df[['Years of Experience', 'Salary', 'Cluster']])


#Here the data is separated into different clusters using KMeans Clustering.





