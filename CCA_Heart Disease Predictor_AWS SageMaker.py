#!/usr/bin/env python
# coding: utf-8

# In[33]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session


# In[13]:


pip install --upgrade pip


# In[2]:


bucket_name = 'myccaprojectpune' 
my_region = boto3.session.Session().region_name 
print(my_region)


# In[3]:


s3 = boto3.resource('s3')
try:
    if  my_region == 'eu-north-1':
        s3.create_bucket(Bucket=bucket_name,CreateBucketConfiguration={'LocationConstraint':'eu-north-1'})
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[4]:


prefix = 'ccamodel'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)    


# In[5]:


import pandas as pd
import numpy as np
import urllib
try:
    urllib.request.urlretrieve ("https://raw.githubusercontent.com/ritzx21/Heart_Disease/main/heart_disease_data.csv", "heart_disease.csv")
    print('Success: downloaded heart_disease.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    df = pd.read_csv('./heart_disease.csv')
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[6]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# In[14]:


get_ipython().system('pip install tensorflow')


# In[28]:


pip install --upgrade pandas numpy


# In[30]:


import numpy as np
import os


# In[24]:


X = df.drop(columns = 'target')
Y = df['target']


# In[31]:


X_train, X_test , Y_train ,Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)


# In[35]:


train_data = pd.concat([X_train, Y_train], axis=1)
test_data = pd.concat([X_test, Y_test], axis=1)

# Save training and testing data to CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# In[36]:


model = LogisticRegression()


# In[37]:


model.fit(X_train , Y_train)


# In[38]:


X_predict_train =  model.predict(X_train)


# In[39]:


accuracy_train = accuracy_score(X_predict_train , Y_train)
print("The accuracy on train data is : ",accuracy_train*100)


# In[40]:


X_predict = model.predict(X_test)


# In[41]:


accuracy = accuracy_score(X_predict , Y_test)


# In[42]:


print("The accuracy of the logistic regression model here is  : ",accuracy*100)


# In[43]:


df.head()


# In[44]:


input_data = (1,0,0,130,157,0,0,160,0,3.6,0,1,2)  # All the features age, sex, cp ,trestbps .... as input tuple

#change input_data to array
input_data_array = np.asarray(input_data)

#reshape the input_data_array in one instance
reshaped = input_data_array.reshape(1,-1)

#predict
prediction = model.predict(reshaped)
print(prediction)

if prediction == 1:
  print("The patient has heart disease")
else:
  print("The patient does not suffer from heart disease")


# In[45]:


import pickle


# In[46]:


filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[47]:


boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'model/heart_disease_model.sav')).upload_file('heart_disease_model.sav')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/model'.format(bucket_name, prefix), content_type='sav')


# In[ ]:




