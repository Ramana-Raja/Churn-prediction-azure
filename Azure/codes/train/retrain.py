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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import logging
    
    logging.basicConfig(
        filename='/var/log/myapp/myapp.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Monitor Training started")
    
    resource = Resource(attributes={"service.name": "python-training-app"})
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
        
        with tracer.start_as_current_span("download-test-data"):
            container_name = "mlopsdata"  
            blob_name_test = "test.parquet"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name_test)
            stream = BytesIO(blob_client.download_blob().readall())
            df_test = pd.read_parquet(stream)
            logging.info("Downloaded df_test for evaluation")
        
        with tracer.start_as_current_span("download-current-model"):
            container_name = "mlopsdata"
            blob_model = "model.pkl"
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_model)
            model_bytes = blob_client.download_blob().readall()
            current_model = joblib.load(BytesIO(model_bytes))
            logging.info("Downloaded current model")
        
        with tracer.start_as_current_span("training-new-model"):
            y_train = df_train['Churn']
            X_train = df_train.drop('Churn', axis=1)
            
            new_model = joblib.loads(joblib.dumps(current_model))  
            new_model.fit(X_train, y_train)
            logging.info("Trained new model")
        
        with tracer.start_as_current_span("ab-testing-evaluation") as eval_span:
            y_test = df_test['Churn']
            X_test = df_test.drop('Churn', axis=1)
            
            current_pred = current_model.predict(X_test)
            new_pred = new_model.predict(X_test)
            
            current_pred_proba = current_model.predict_proba(X_test)[:, 1] if hasattr(current_model, 'predict_proba') else None
            new_pred_proba = new_model.predict_proba(X_test)[:, 1] if hasattr(new_model, 'predict_proba') else None
            
            def calculate_metrics(y_true, y_pred, y_pred_proba=None):
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1': f1_score(y_true, y_pred, average='weighted')
                }
                if y_pred_proba is not None:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                return metrics
            
            current_metrics = calculate_metrics(y_test, current_pred, current_pred_proba)
            new_metrics = calculate_metrics(y_test, new_pred, new_pred_proba)
            
            logging.info(f"Current model test metrics: {current_metrics}")
            logging.info(f"New model test metrics: {new_metrics}")
            
            for metric, value in current_metrics.items():
                eval_span.set_attribute(f"current_model.test_{metric}", value)
            for metric, value in new_metrics.items():
                eval_span.set_attribute(f"new_model.test_{metric}", value)

            primary_metric = 'auc'  
            improvement_threshold = 0.01  
            
            if new_metrics[primary_metric] > current_metrics[primary_metric] + improvement_threshold:
                model_improved = True
                improvement = new_metrics[primary_metric] - current_metrics[primary_metric]
                logging.info(f"New model shows improvement! Test {primary_metric} increased by {improvement:.4f}")
                eval_span.set_attribute("model_deployment_decision", "deploy")
            else:
                model_improved = False
                logging.info(f"New model does not show sufficient improvement on test data. Keeping current model.")
                eval_span.set_attribute("model_deployment_decision", "keep_current")
        

        if model_improved:
            with tracer.start_as_current_span("save-improved-model"):
                buffer = BytesIO()
                joblib.dump(new_model, buffer)
                buffer.seek(0)
                blob_name = "model.pkl"
                container_name = "mlopsdata"
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_client.upload_blob(buffer, overwrite=True)
                
                parquet_buffer_train = BytesIO()
                df_train.to_parquet(parquet_buffer_train)
                parquet_buffer_train.seek(0)
                blob_name = "train.parquet"
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_client.upload_blob(parquet_buffer_train, overwrite=True)
                
                import datetime
                metrics_df = pd.DataFrame({
                    'timestamp': [datetime.datetime.now(), datetime.datetime.now()],
                    'model_version': ['current', 'new'],
                    'test_accuracy': [current_metrics['accuracy'], new_metrics['accuracy']],
                    'test_precision': [current_metrics['precision'], new_metrics['precision']],
                    'test_recall': [current_metrics['recall'], new_metrics['recall']],
                    'test_f1': [current_metrics['f1'], new_metrics['f1']],
                    'test_auc': [current_metrics.get('auc', None), new_metrics.get('auc', None)]
                })
                
                metrics_buffer = BytesIO()
                metrics_df.to_parquet(metrics_buffer)
                metrics_buffer.seek(0)
                blob_client = blob_service_client.get_blob_client(container="mlopsdata", blob="model_test_metrics.parquet")
                blob_client.upload_blob(metrics_buffer, overwrite=True)
                
                logging.info("New model saved successfully after test evaluation")
        else:
            logging.info("Model not updated - no significant improvement detected on test data")
    
    logging.info("Monitor Training finished")
