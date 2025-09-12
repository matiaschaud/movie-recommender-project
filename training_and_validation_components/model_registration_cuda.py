from kfp.dsl import component


@component(base_image="matichaud/movie-recommender:v1")
def promote_model_to_staging_cuda(
        model_run_id: str,
        registered_model_name: str,
        rms_threshold: float,
        precision_threshold: float,
        top_k: int,
        recall_threshold: float,
        AWS_ACCESS_KEY_ID:str, 
        AWS_SECRET_ACCESS_KEY:str,
        MLFLOW_S3_ENDPOINT_URL:str,
        mlflow_uri: str):

    import mlflow.pytorch
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    import os
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
    
    mlflow.set_tracking_uri(uri=mlflow_uri)
    client = MlflowClient()

    current_staging = None
    try:
        current_staging = client.get_model_version_by_alias(registered_model_name, "staging")
    except RestException:
        print("No staging model found. Auto upgrade current run to staging.")
    
    if current_staging is not None:
        # Check if the model we're trying to promote is already the current staging model.
        if current_staging.run_id == model_run_id:
            print("Input run is already the current staging.")
            return
            
        current_staging_model_data = client.get_run(current_staging.run_id).data.to_dictionary()
        staging_model_metrics = current_staging_model_data['metrics']

        new_model_data = client.get_run(model_run_id).data.to_dictionary()
        new_model_metrics = new_model_data['metrics']

        if (new_model_metrics['rms'] - staging_model_metrics['rms']) > rms_threshold:
            return

        if (new_model_metrics[f'precision_{top_k}'] - staging_model_metrics[f'precision_{top_k}']) < precision_threshold:
            return

        if (new_model_metrics[f'recall_{top_k}'] - staging_model_metrics[f'recall_{top_k}']) < recall_threshold:
            return

    result = mlflow.register_model(f"runs:/{model_run_id}/model", "recommender_production")
    client.set_registered_model_alias("recommender_production", "staging", result.version)
