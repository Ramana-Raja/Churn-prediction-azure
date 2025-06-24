def retrain():
    import pandas as pd
    import joblib
    import os
    from azure.storage.blob import BlobServiceClient
    from io import BytesIO
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    import logging

    logging.basicConfig(
        filename='/var/log/myapp/myapp.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info("Monitor Training started")

    resource = Resource(attributes={
        "service.name": "python-training-app"
    })

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    otlp_exporter = OTLPSpanExporter(
        endpoint="http://tempo.monitoring.svc.cluster.local:4317",
        insecure=True
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("model-retraining") as span:
        storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_account_key = os.getenv("AZURE_STORAGE_KEY")

        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        with tracer.start_as_current_span("download-train-data"):
            container_name = "model-retrain"
            blob_name_train = "train.parquet"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name_train)
            stream = BytesIO(blob_client.download_blob().readall())
            df_train = pd.read_parquet(stream)
            logging.info("Downloaded df_train")

        with tracer.start_as_current_span("download-model"):
            container_name = "mlopsdata"
            blob_model = "model.pkl"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_model)
            model_bytes = blob_client.download_blob().readall()
            model = joblib.load(BytesIO(model_bytes))
            logging.info("Downloaded model")

        with tracer.start_as_current_span("training-model"):

            y = df_train['Churn']
            x = df_train.drop('Churn', axis=1)

            model.fit(x,y)

            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            blob_name = "model.pkl"
            container_name = "mlopsdata"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            blob_client.upload_blob(buffer, overwrite=True)


            parquet_buffer_train = BytesIO()
            df_train.to_parquet(parquet_buffer_train)
            parquet_buffer_train.seek(0)

            blob_name = "train.parquet"
            container_name = "mlopsdata"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            blob_client.upload_blob(parquet_buffer_train, overwrite=True)

    logging.info("Monitor Training finished")

