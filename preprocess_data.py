import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

df = pd.read_csv("dataset.csv")

df = df.drop('customerID',axis=1)

df['gender'] = np.where(df['gender']=='Male',1,0)
df['InternetService'] = df['InternetService'].map({'DSL':1 , 'Fiber optic':2,'No':0 })
df['MultipleLines'] = df['MultipleLines'].map({'No phone service':-1 , 'Yes':1,'No':0 })

no_internet_service = ['OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                       'StreamingMovies']
for col in no_internet_service:
    df[col] = df[col].map({'No internet service':-1 , 'Yes':1,'No':0 })


df['Contract'] = df['Contract'].map({'Month-to-month':1 , 'One year':10,'Two year':20 })
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check':1 , 'Mailed check':2,'Bank transfer (automatic)':3
                                               ,'Credit card (automatic)':4})

df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

yes_no_cols = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'
]
for col in yes_no_cols:
    df[col] = np.where(df[col] == 'Yes', 1, 0)



corr = df.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.show()