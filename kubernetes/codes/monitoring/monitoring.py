import time
import pandas as pd
from evidently import Report, DataDefinition, Dataset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
import joblib
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from prometheus_client import Gauge, start_http_server

drift_value = Gauge('data_drift_value', 'Drift value of Churn')
drifted_columns_count = Gauge('drifted_columns_count', 'Number of drifted columns')
missing_value_count = Gauge('missing_value_count', 'Missing value count in Churn')

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
    stream = BytesIO(blob_client.download_blob().readall())
    df_train = pd.read_parquet(stream)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name_test)
    stream = BytesIO(blob_client.download_blob().readall())
    df_test = pd.read_parquet(stream)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_model)
    model_bytes = blob_client.download_blob().readall()
    model = joblib.load(BytesIO(model_bytes))

    df_test = df_test.drop('Churn', axis=1)
    df_test['Churn'] = model.predict(df_test)

    data_definition = DataDefinition(
        numerical_columns=list(df_train.columns),
        categorical_columns=[],
    )

    current_dataset = Dataset.from_pandas(df_test, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(df_train, data_definition=data_definition)

    report = Report(metrics=[
        ValueDrift(column='Churn'),
        DriftedColumnsCount(),
        MissingValueCount(column='Churn'),
    ])

    run = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result = run.dict()

    drift_value.set(result['metrics'][0]['value'])
    drifted_columns_count.set(result['metrics'][1]['value']['count'])
    missing_value_count.set(result['metrics'][2]['value']['count'])

if __name__ == "__main__":

    start_http_server(8000)

    while True:
        monitor()
        print("Drift check completed. Next check in 5 hours.")
        time.sleep(5 * 60 )