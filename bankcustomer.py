
#BANK CUSTOMER CHURN

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Import the dataset
df = pd.read_csv("D:/DATASCIENCE/MachineLearningWorks/Bank Customer Churn Prediction.csv")
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

#Checking for duplicate values
print(df.duplicated())

#Sum of duplicate values
print(df.duplicated().sum())

#To find unique values
print(df.nunique())

#Information about the data
print(df.info())

#Shape of dataset
print(df.shape)

# Separate the data into independent and dependent variables
x = df.drop('churn', axis=1)
y = df['churn']
print(x)
print(y)

#Shape of x and y
print("x shape:",x.shape)
print("y shape:",y.shape)

#Change the country column into numeric
label_encoder = LabelEncoder()
x['country'] = label_encoder.fit_transform(x['country'])
print(x['country'])

# Store the mapping of original labels to encoded values
type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(type_mapping)

#Change the gender column into numeric
label_encoder = LabelEncoder()
x['gender'] = label_encoder.fit_transform(x['gender'])
print(x['gender'])

# Store the mapping of original labels to encoded values
type_mapping1 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(type_mapping1)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Print train and test data
print("x_train",x_train)
print("y_train",y_train)
print("x_test",x_test)
print("y_test",y_test)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

#Feature Scaling
#Support Vector Classifier
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

#Fitting Support Vector Classifier to the training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Result-Y":y_test,"PredictionResult":y_pred})
print(df2.to_string())

#Evaluating the Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


# While testing this data we got 80.35% accuracy using Support Vector Classifier.
