import kfp
from kfp import dsl

from data_components import (
    download_ml25m_data,
    unzip_data,
    csv_to_parquet,
    split_dataset,
    put_to_minio,
    qa_data,
    load_to_postgres,  # Import the new component
)

@dsl.pipeline(
    name='Data prep pipeline',
    description='A pipeline that retrieves data from movielens and ingests it into parquet files and a PostgreSQL database'
)
def dataprep_pipeline(
    minio_bucket: str = 'datasets',
    random_init: int = 42,
    db_host: str = 'postgres-postgresql.feast.svc.cluster.local',
    db_name: str = '',
    db_user: str = '',
    db_port: int = 5432,
    db_password: str = '',
):
    download_dataset = download_ml25m_data()
    unzip_folder = unzip_data(input_path=download_dataset.outputs['output_path_one'])

    # New step to load data into PostgreSQL
    load_to_postgres_task = load_to_postgres(
        ratings_input_path=unzip_folder.outputs['ratings_output_path'],
        movies_input_path=unzip_folder.outputs['movies_output_path'],
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        db_port=db_port,
    )

    # Subsequent tasks will continue as before, using the unzipped artifacts
    # The `ratings_parquet_op` and `movies_parquet_op` tasks will now run in parallel
    # with the `load_to_postgres_task` as they share the same input
    ratings_parquet_op = csv_to_parquet(inputFile=unzip_folder.outputs['ratings_output_path'])
    movies_parquet_op = csv_to_parquet(inputFile=unzip_folder.outputs['movies_output_path'])
    split_op = split_dataset(input_parquet=ratings_parquet_op.output, random_state=random_init)
    u1 = put_to_minio(inputFile=movies_parquet_op.output, upload_file_name='movies.parquet.gzip', bucket=minio_bucket)
    u2 = put_to_minio(inputFile=split_op.output, bucket=minio_bucket)
    qa_op = qa_data(bucket=minio_bucket).after(u2)
    
    # You can add a dependency to ensure the loading finishes before any subsequent steps
    # that might rely on the data being in the DB.
    # For this pipeline, it's not strictly necessary since the Parquet conversion still uses the artifacts,
    # but it's a good practice.
    ratings_parquet_op.after(load_to_postgres_task)
    movies_parquet_op.after(load_to_postgres_task)

    # Set caching options to False for all new tasks
    download_dataset.set_caching_options(True)
    unzip_folder.set_caching_options(True)
    load_to_postgres_task.set_caching_options(False) # New task
    ratings_parquet_op.set_caching_options(False)
    movies_parquet_op.set_caching_options(False)
    split_op.set_caching_options(False)
    u1.set_caching_options(False)
    u2.set_caching_options(False)
    qa_op.set_caching_options(False)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=dataprep_pipeline,
        package_path='compiled_pipelines/dataPrep_pipeline.yaml'
    )