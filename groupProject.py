#!/usr/bin/env python
# coding: utf-8

"""

Student Name: Jack Fitzgerald
Class Group: CS-3
Student ID: R00205373

"""

# Importing libraries
import numpy as np
import pandas as pd
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import firebase_admin
from firebase_admin import credentials, firestore

# Initialise connection
cred = credentials.Certificate("/Users/jackfitzgerald/jupyterNotebooks/sjaa-3rd-year-group-project-firebase-adminsdk-6ooya-453c2c9917.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firestore collection
collectionRef = db.collection("users")

# Loading the dataset for different health conditions
heartDataset = loadtxt('heart_failure_clinical_records_dataset.csv', delimiter=',', skiprows=1)
lungDataset = loadtxt('survey_lung_cancer.csv', delimiter=',', skiprows=1)
strokeDataset = loadtxt('healthcare-dataset-stroke-data.csv', delimiter=',', skiprows=1)

# Calculating feature averages
heartDf = pd.read_csv('heart_failure_clinical_records_dataset.csv')
lungDf = pd.read_csv('survey_lung_cancer.csv')
strokeDf = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Calculate feature averages for imputation
heartFeatureAverage = heartDf.mean()
lungFeatureAverage = lungDf.mean()
strokeFeatureAverage = strokeDf.mean()

# Defining  heart disease prediction model
heartModel = Sequential()
heartModel.add(Dense(12, input_shape=(12,), activation='relu'))
heartModel.add(Dense(12, activation='relu'))
heartModel.add(Dense(1, activation='sigmoid'))
heartModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Indexing and slicing array
# Colon before comma - we want to include all rows
# 0:12 after comma - columns we want to include
X = heartDataset[:, 0:12]
y = heartDataset[:, 12]

# Train heart model
heartModel.fit(X, y, epochs=300, batch_size=300)
_, accuracy = heartModel.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# Defining lung cancer prediction model
lungModel = Sequential()
lungModel.add(Dense(14, input_shape=(14,), activation='relu'))
lungModel.add(Dense(14, activation='relu'))
lungModel.add(Dense(1, activation='sigmoid'))
lungModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Indexing and slicing array
# Colon before comma - we want to include all rows
# 0:14 after comma - columns we want to include
X = lungDataset[:, 0:14]
y = lungDataset[:, 14]

# Train lung model
lungModel.fit(X, y, epochs=310, batch_size=310)
_, accuracy = lungModel.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# Defining stroke prediction model
strokeModel = Sequential()
strokeModel.add(Dense(8, input_shape=(8,), activation='relu'))
strokeModel.add(Dense(8, activation='relu'))
strokeModel.add(Dense(1, activation='sigmoid'))
strokeModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Indexing and slicing array
# Colon before comma - we want to include all rows
# 0:8 after comma - columns we want to include
X = strokeDataset[:, 0:8]
y = strokeDataset[:, 8]

# Train stroke model
strokeModel.fit(X, y, epochs=500, batch_size=500)
_, accuracy = strokeModel.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# Function for making predictions based on user data and models
def makePrediction(userData, heartModel, lungModel, strokeModel, heartThreshold=0.4, lungThreshold=0.6, strokeThreshold=0.3):
    try:
        # Extracting the name from the userData
        firstName = userData.get('first name')
        lastName = userData.get('last name')
        
        # Printing userData for debugging
        print('User Data: ', userData)
    
        # Defining the features for each model
        heart_features = ['age','anaemia','creatinine','diabetes','ejection','highblood','platelets','serumcreatinine','serumsodium','sex','smoking','time']
        lung_features = ['gender','age','smoking','yellowfingers','anxiety','chronic disease','fatigue','allergy','wheezing','alcohol','choughing','shortbreath','swallow','chestpain']
        stroke_features = ['gender','age','hypertension','heartdisease','married','glucose','bmi','smoking']
    
        # Extract features for mode and predict conditions and thresholds
        heartFeaturesArray = np.array([[userData.get(f, 0) for f in heart_features]], dtype=np.float32)
        heartPrediction = (heartModel.predict(heartFeaturesArray) > heartThreshold).astype(int)

        lungFeaturesArray = np.array([[userData.get(f, 0) for f in lung_features]], dtype=np.float32)
        lungPrediction = (lungModel.predict(lungFeaturesArray) > lungThreshold).astype(int)

        strokeFeaturesArray = np.array([[userData.get(f, 0) for f in stroke_features]], dtype=np.float32)
        strokePrediction = (strokeModel.predict(strokeFeaturesArray) > strokeThreshold).astype(int)

        # Raw prediction for each model
        heartProb = heartModel.predict(heartFeaturesArray)[0]
        lungProb = lungModel.predict(lungFeaturesArray)[0]
        strokeProb = strokeModel.predict(strokeFeaturesArray)[0]
    
        # List to store results
        predictions = []
    
        # Check predictions against thresholds and append results with confidence scores
        if heartProb > heartThreshold:
            predictions.append(f'Heart Disease (Confidence: {heartProb[0]:.2})')
        if lungProb > lungThreshold:
            predictions.append(f'Lung Cancer (Confidence: {lungProb[0]:.2f})')
        if strokeProb > strokeThreshold:
            predictions.append(f'Stroke (Confidence: {strokeProb[0]:.2f})')
        
        # Append condition to list if positive
        if heartPrediction[0] == 1:
            predictions.append("Heart Disease")
        
        if lungPrediction[0] == 1:
            predictions.append("Lung Cancer")
    
        if strokePrediction[0] == 1:
            predictions.append("Stroke")
        
        # Final result print
        if predictions:
            return f'{firstName} {lastName} has predictions for: {", ".join(predictions)}'
        else:
            return f'{firstName} {lastName} has no predictions for the given conditions'
    # Error handling, returns error message if exception occurs
    except Exception as e:
        return f'An error occurred during prediction: {str(e)}'


# In[ ]:





# In[72]:


for doc in collectionRef.stream():
    # convert doc to dictionary. 
    userData = doc.to_dict()
    # calling prediction function with data and models
    predictions = makePrediction(userData, heartModel, lungModel, strokeModel)
    # Print user id, predictions
    print(f"User ID: {doc.id}\n{predictions}\n\n")


# In[ ]:




