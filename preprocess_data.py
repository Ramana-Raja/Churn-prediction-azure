import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO, StringIO
from azure.storage.blob import BlobServiceClient

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


def plot_corr():
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()
df_no_index = df.reset_index(drop=True)

parquet_buffer = BytesIO()
df_no_index .to_parquet(parquet_buffer)
parquet_buffer.seek(0)

storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
storage_account_key = os.getenv("AZURE_STORAGE_KEY")

connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = "mlopsdata"
blob_name = "processed_data.parquet"

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)