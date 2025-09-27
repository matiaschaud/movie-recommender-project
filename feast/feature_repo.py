# This file is a skeleton for a feature repository.

from feast import FeatureView, Field, Entity, ValueType, Project, FeatureService
# IMPORT CHANGE 1: Replace PostgreSQLSource import with BigQuerySource
from feast.infra.offline_stores.bigquery import BigQuerySource
from feast.types import Int64, Float64, String
from datetime import timedelta


# Define a project for the feature repo
project = Project(name="movie_recommender", description="Repo for the Movie recommender Tesis project")

# Define an entity for each unique identifier in your data.
# An entity serves as the primary key for looking up features.
user_id = Entity(name="userid", description="User ID", value_type=ValueType.INT64)
movie_id = Entity(name="movieid", description="Movie ID", value_type=ValueType.INT64)

# Create a data source for your ratings data in BigQuery.
# This source defines where Feast will find the raw data.
# CLASS CHANGE 1: Use BigQuerySource instead of PostgreSQLSource
ratings_source = BigQuerySource(
    # The name of the data source.
    name="ratings_source",
    # The SQL query that extracts the ratings data from your database.
    # We use BigQuery's built-in functions for timestamp conversion.
    # NOTE: You must use the fully qualified table name (e.g., 'project.dataset.table')
    # or ensure your BQ dataset is configured in the feature_store.yaml.
    query="""
        SELECT 
            CAST(userid AS INT64) AS userid,
            CAST(movieid AS INT64) AS movieid, 
            CAST(rating AS FLOAT64) AS rating, 
            # Use BigQuery's TIMESTAMP_SECONDS or TIMESTAMP_MILLIS conversion 
            # assuming your 'timestamp' column is an epoch integer.
            TIMESTAMP_SECONDS(timestamp) as event_timestamp 
        FROM 
            # REPLACE_THIS_WITH_YOUR_RATINGS_TABLE
            `tesis-master-ciencia-de-datos.feast_staging.ratings`
    """,
    # The column that contains the timestamp for each event.
    timestamp_field="event_timestamp",
)

# Create a data source for your movies data.
# CLASS CHANGE 2: Use BigQuerySource instead of PostgreSQLSource
movies_source = BigQuerySource(
    # The name of the data source.
    name="movies_source",
    # This query now reuses the created_at column and aliasing it to event_timestamp
    # so that both feature sources use the same timestamp column name
    # We use a hardcoded TIMESTAMP literal for BigQuery.
    query="""
        SELECT 
            CAST(movieid AS INT64) AS movieid, 
            title, 
            genres, 
            CAST('2000-01-01 00:00:00' AS TIMESTAMP) as event_timestamp 
        FROM 
            # REPLACE_THIS_WITH_YOUR_MOVIES_TABLE
            `tesis-master-ciencia-de-datos.feast_staging.movies`
    """,
    timestamp_field="event_timestamp",
)

# Define a FeatureView for the user ratings data.
# This view links the raw data to the entities and defines the features available.
user_rating_feature_view = FeatureView(
    name="user_rating_feature_view",
    entities=[user_id, movie_id],
    # ttl=timedelta(days=365), # A Time to Live for the features.
    source=ratings_source,
    schema=[
        Field(name="rating", dtype=Float64),
    ],
)

# Define a FeatureView for the movie details.
movie_details_feature_view = FeatureView(
    name="movie_details_feature_view",
    entities=[movie_id],
    # ttl=timedelta(days=365),
    source=movies_source,
    schema=[
        Field(name="title", dtype=String),
        Field(name="genres", dtype=String),
    ],
)

# Define a FeatureService to group features for retrieval.
movie_recommender_service_all = FeatureService(
    name="movie_recommender_service_all",
    features=[
        user_rating_feature_view,
        movie_details_feature_view,
    ],
)