
#CREDIT RISK

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Import the dataset
df = pd.read_csv("D:/DATASCIENCE/MachineLearningWorks/CreditRisk.csv")
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

#Change the Credit_Risk column into numeric
label_encoder = LabelEncoder()
df['Credit_Risk'] = label_encoder.fit_transform(df['Credit_Risk'])
print(df['Credit_Risk'])

# Store the mapping of original labels to encoded values
type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(type_mapping)

# Separate the data into independent and dependent variables
x = df.drop('Credit_Risk', axis=1)
y = df['Credit_Risk']
print(x)
print(y)

#Shape of x and y
print("x shape:",x.shape)
print("y shape:",y.shape)

#Train and test split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

#Print train and test data
print("x_train",x_train)
print("y_train",y_train)
print("x_test",x_test)
print("y_test",y_test)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Fitting DecisionTree Classifier to the training set
classifier= DecisionTreeClassifier()
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred = classifier.predict(x_test)
df2=pd.DataFrame({"Actual Result-Y":y_test,"PredictionResult":y_pred})
print(df2)

#Evaluating the Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


# While testing this data we got 96.32% accuracy using Decision Tree Classifier.