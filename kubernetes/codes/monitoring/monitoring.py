import time

import pandas as pd

from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
import joblib
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO

def monitor():
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    storage_account_key = os.getenv("AZURE_STORAGE_KEY")

    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    container_name = "mlopsdata"
    blob_name_train = "train.parquet"
    blob_name_test = "test.parquet"
    blob_model = "model.pkl"


    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name_train)

    download_stream = BytesIO()
    download_stream.write(blob_client.download_blob().readall())
    download_stream.seek(0)

    df_train = pd.read_parquet(download_stream)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name_test)

    download_stream = BytesIO()
    download_stream.write(blob_client.download_blob().readall())
    download_stream.seek(0)
    df_test = pd.read_parquet(download_stream)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_model)

    model_bytes = blob_client.download_blob().readall()
    buffer = BytesIO(model_bytes)

    model = joblib.load(buffer)


    df_test = df_test.drop('Churn',axis=1)

    df_test['Churn'] = model.predict(df_test)


    data_definition = DataDefinition(
        numerical_columns=list(df_train.columns),
        categorical_columns=[],
    )

    current_dataset = Dataset.from_pandas(df_test, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(df_train, data_definition=data_definition)

    report = Report(metrics = [
        ValueDrift(column='Churn'),
        DriftedColumnsCount(),
        MissingValueCount(column='Churn'),
    ])

    run = report.run(reference_data=reference_dataset, current_data=current_dataset)

    result = run.dict()
    print(result['metrics'][0]['value'])
    print(result['metrics'][1]['value']['count'])
    print(result['metrics'][2]['value']['share'])

if __name__ == "__main__":
    while True:
        print("new thing")
        monitor()
        time.sleep(5)
