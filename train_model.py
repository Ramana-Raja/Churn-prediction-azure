from azure.storage.blob import BlobServiceClient
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
import joblib


storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
storage_account_key = os.getenv("AZURE_STORAGE_KEY")

connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = "mlopsdata"
blob_name = "processed_data.parquet"

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

download_stream = BytesIO()
download_stream.write(blob_client.download_blob().readall())
download_stream.seek(0)

df = pd.read_parquet(download_stream)

x = df.drop("Churn",axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best_param = {
    'max_depth': 8,
    'n_estimators': 10,
    'min_samples_split': 4,
    'criterion': "log_loss",
    'min_samples_leaf': 4,
}

model = RandomForestClassifier(**best_param, random_state=42)
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

buffer = BytesIO()
joblib.dump(model,buffer)
buffer.seek(0)

blob_name = "model.pkl"

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
blob_client.upload_blob(buffer, overwrite=True)