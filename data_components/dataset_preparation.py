from kfp import dsl
from kfp.dsl import Input, Output, Artifact


@dsl.component(base_image="python:3.11", packages_to_install=["requests"])
def download_ml25m_data(output_path_one: Output[Artifact]):
    import requests
    url = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    response = requests.get(url, stream=True, verify=False, timeout=60)
    print(output_path_one.path)
    with open(output_path_one.path, 'wb') as file: 
        for chunk in response.iter_content(chunk_size=1024*1024):  # D
            if chunk:
                file.write(chunk)


@dsl.component(base_image="python:3.11")            
def unzip_data(input_path: Input[Artifact], ratings_output_path: Output[Artifact], movies_output_path: Output[Artifact]):
    import zipfile

    with zipfile.ZipFile(input_path.path, 'r') as z:
        with open(ratings_output_path.path, 'wb') as f:
            f.write(z.read('ml-25m/ratings.csv'))
        with open(movies_output_path.path, 'wb') as f:
            f.write(z.read('ml-25m/movies.csv'))


@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def split_dataset(input_parquet: Input[Artifact], dataset_path: Output[Artifact], random_state: int = 42):
    from sklearn.model_selection import train_test_split
    import os
    import pandas as pd
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    ratings_df = pd.read_parquet(input_parquet.path)

    # train is now 75% of the entire data set
    train, test = train_test_split(
        ratings_df,                                    
        test_size=1 - train_ratio,
        random_state=random_state)

    n_users = ratings_df.userId.max()
    n_items = ratings_df.movieId.max()

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    val, test = train_test_split(
        test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state)
    os.mkdir(dataset_path.path)
    train.to_parquet(os.path.join(dataset_path.path, 'train.parquet.gzip'), compression='gzip')
    test.to_parquet(os.path.join(dataset_path.path, 'test.parquet.gzip'), compression='gzip')
    val.to_parquet(os.path.join(dataset_path.path, 'val.parquet.gzip'), compression='gzip')


@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def csv_to_parquet(inputFile: Input[Artifact], output_path: Output[Artifact]):
    import pandas as pd
    df = pd.read_csv(inputFile.path, index_col=False)
    df.to_parquet(output_path.path, compression='gzip') 


@dsl.component(base_image="python:3.11", packages_to_install=["boto3"])
def put_to_minio(inputFile: Input[Artifact], upload_file_name:str='', bucket: str='datasets'):
    import boto3
    import os
    minio_client = boto3.client(                          
        's3',                                              
        endpoint_url='http://minio-service.kubeflow:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123')
    try:
        minio_client.create_bucket(Bucket=bucket)
    except minio_client.exceptions.BucketAlreadyExists:
        # Bucket already created.
        print(f"{bucket} already exists, not creating it.")
    except minio_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"{bucket} already exists, not creating it.")
    
        
    if os.path.isdir(inputFile.path):
        for file in os.listdir(inputFile.path):
            s3_path = os.path.join('ml-25m', file)
            minio_client.upload_file(os.path.join(inputFile.path, file), bucket, s3_path)
    else:
        if upload_file_name == '':
            _, file = os.path.split(inputFile.path)
        else:
            file = upload_file_name
        s3_path = os.path.join('ml-25m', file)
        minio_client.upload_file(inputFile.path, bucket, s3_path)


@dsl.component(base_image="python:3.11", packages_to_install=["pyarrow", "pandas"]) 
def qa_data(bucket: str = 'datasets', dataset: str = 'ml-25m'):
    from pyarrow import fs, parquet
    print("Running QA")
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key='minio',
        secret_key='minio123',
        scheme='http')
    train_parquet = minio.open_input_file(f'{bucket}/{dataset}/train.parquet.gzip')
    df = parquet.read_table(train_parquet).to_pandas()
    assert df.shape[1] == 4
    assert df.shape[0] >= 0.75 * 25 * 1e6
    print('QA passed!')


@dsl.component(base_image="python:3.11", packages_to_install=["psycopg2-binary"])
def load_to_postgres(
    ratings_input_path: Input[Artifact],
    movies_input_path: Input[Artifact],
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    db_port: int = 5432,
    
):
    import psycopg2
    from datetime import datetime, timezone

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()

        # Create tables with primary keys, foreign keys, and indexes
        print("Creating 'movies' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movieId INTEGER PRIMARY KEY,
                title VARCHAR(255),
                genres VARCHAR(255),
                created_at timestamp DEFAULT current_timestamp
            );
        """)

        print("Creating 'ratings' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                userId INTEGER,
                movieId INTEGER,
                rating NUMERIC(3, 1),
                "timestamp" BIGINT,
                PRIMARY KEY (userId, movieId),
                CONSTRAINT fk_movie
                    FOREIGN KEY(movieId) 
                    REFERENCES movies(movieId)
            );
        """)

        print("Creating index on 'ratings.userId'...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ratings_userId ON ratings (userId);")

        print("Truncating tables to prevent duplicates...")
        cur.execute("TRUNCATE TABLE movies CASCADE;")

        print("Loading movies data...")
        with open(movies_input_path.path, 'r') as f_movies:
            cur.copy_expert("COPY movies(movieId, title, genres) FROM STDIN WITH (FORMAT CSV, HEADER)", f_movies)

        print("Loading ratings data...")
        with open(ratings_input_path.path, 'r') as f_ratings:
            cur.copy_expert("COPY ratings FROM STDIN WITH (FORMAT CSV, HEADER)", f_ratings)

        conn.commit()
        print("Data loaded successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

from kfp.dsl import component, Input, Artifact

from kfp.dsl import component, Input, Artifact

# We need the base image to have both pandas and the BQ library installed.
@component(
    base_image="python:3.11", 
    packages_to_install=["google-cloud-bigquery", "pandas"] 
)
def load_to_bigquery(
    ratings_input_path: Input[Artifact],
    movies_input_path: Input[Artifact],
    project_id: str,
    dataset_id: str,
    # New argument to receive the service account JSON key as a string
    service_account_json: str, 
    ratings_table_name: str = "ratings",
    movies_table_name: str = "movies",
):
    import os
    import tempfile
    from google.cloud import bigquery
    from google.oauth2 import service_account
    
    print("--- Configuring GCP Credentials ---")
    
    # 1. Create a temporary file to store the credentials
    # The 'delete=False' is important to keep the file until the component finishes
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_key_file:
        temp_key_file.write(service_account_json)
        credentials_path = temp_key_file.name

    print(f"Service Account key saved to temporary path: {credentials_path}")

    # 2. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
    # This enables Application Default Credentials (ADC) to automatically find the key.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Alternative Method (Explicit Credentials):
    # credentials_info = json.loads(service_account_json)
    # credentials = service_account.Credentials.from_service_account_info(credentials_info)
    # client = bigquery.Client(project=project_id, credentials=credentials)
    
    # 3. Initialize the BigQuery client (ADC will automatically use the environment variable)
    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        # Crucial for debugging: If authentication fails, this prints the error
        print("ERROR: Failed to initialize BigQuery client using provided credentials.")
        print("Please check the JSON key and IAM permissions.")
        raise e
        
    print(f"BigQuery Client authenticated and ready.")
    
    # Clean up the temporary file (optional, as the container is destroyed after the component runs)
    # os.remove(credentials_path)
    
    # ----------------------------------------------------------------------
    # START BIGQUERY LOAD LOGIC
    # ----------------------------------------------------------------------

    ratings_table_ref = f"{project_id}.{dataset_id}.{ratings_table_name}"
    movies_table_ref = f"{project_id}.{dataset_id}.{movies_table_name}"
    
    print(f"Targeting BigQuery Project: {project_id}, Dataset: {dataset_id}")

    # 1. MOVIES TABLE LOAD (Simplified snippet for brevity)
    print(f"Starting load for movies data to table: {movies_table_ref}")
    
    movies_schema = [
        bigquery.SchemaField("movieId", "INT64"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("genres", "STRING"),
    ]
    
    movies_job_config = bigquery.LoadJobConfig(
        schema=movies_schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    with open(movies_input_path.path, "rb") as source_file:
        movies_job = client.load_table_from_file(
            source_file,
            movies_table_ref,
            job_config=movies_job_config,
        ) 
    movies_job.result() 
    print(f"Movies data loaded successfully to {movies_table_ref}. Rows: {movies_job.output_rows}")
    
    # 2. RATINGS TABLE LOAD (Simplified snippet for brevity)
    print(f"Starting load for ratings data to table: {ratings_table_ref}")

    ratings_schema = [
        bigquery.SchemaField("userId", "INT64"),
        bigquery.SchemaField("movieId", "INT64"),
        bigquery.SchemaField("rating", "FLOAT64"),
        bigquery.SchemaField("timestamp", "INT64"),
    ]
    
    ratings_job_config = bigquery.LoadJobConfig(
        schema=ratings_schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    with open(ratings_input_path.path, "rb") as source_file:
        ratings_job = client.load_table_from_file(
            source_file,
            ratings_table_ref,
            job_config=ratings_job_config,
        ) 

    ratings_job.result() 
    print(f"Ratings data loaded successfully to {ratings_table_ref}. Rows: {ratings_job.output_rows}")