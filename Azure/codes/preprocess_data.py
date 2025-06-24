import pandas as pd
import numpy as np
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split

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


df_no_index = df.reset_index(drop=True)


df_train, df_test = train_test_split(df_no_index, test_size=0.2, random_state=42)


parquet_buffer_train = BytesIO()
df_train.to_parquet(parquet_buffer_train)  # pass the BytesIO buffer here
parquet_buffer_train.seek(0)               # reset buffer pointer to start

parquet_buffer_test = BytesIO()
df_test.to_parquet(parquet_buffer_test)
parquet_buffer_test.seek(0)

storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
storage_account_key = os.getenv("AZURE_STORAGE_KEY")

connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = "mlopsdata"
blob_name = "train.parquet"

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
blob_client.upload_blob(parquet_buffer_train.getvalue(), overwrite=True)

container_name = "mlopsdata"
blob_name = "test.parquet"

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
blob_client.upload_blob(parquet_buffer_test.getvalue(), overwrite=True)