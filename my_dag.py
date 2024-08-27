from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator

from datetime import datetime
from pymongo import MongoClient
import os
import certifi
import pandas as pd
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType

# Define the default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 8, 16),
    'retries': 1,
}

# Initialize Spark session
spark = SparkSession.builder \
    .appName("HotelRecommendation2") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.shuffle.file.buffer", "32k") \
    .config("spark.shuffle.memoryFraction", "0.3") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.rpc.message.maxSize", "256") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

spark.catalog.clearCache()

def calculate_weights(df):
    # Count the number of reviews for each user and item
    user_counts = df.groupBy("numeric_id").count().withColumnRenamed("count", "user_count")
    item_counts = df.groupBy("offering_id").count().withColumnRenamed("count", "item_count")
    
    # Join counts with the original DataFrame
    df = df.join(user_counts, on="numeric_id", how="left")
    df = df.join(item_counts, on="offering_id", how="left")
    
    # Define a function to calculate weight based on counts
    def compute_weight(user_count, item_count):
        # Example weight calculation: inverse of counts (you can customize this)
        user_weight = 1 / (user_count + 1)
        item_weight = 1 / (item_count + 1)
        return user_weight * item_weight
    
    weight_udf = udf(compute_weight, FloatType())
    df = df.withColumn("weight", weight_udf(col("user_count"), col("item_count")))
    
    return df

def extract_and_train():
    # MongoDB connection URI
    mongo_uri = ''

    # Use certifi to get the path to the CA bundle
    ca = certifi.where()

    # Initialize the MongoClient with TLS settings
    client = MongoClient(
        mongo_uri,
        tls=True,  # Enable TLS
        tlsCAFile=ca  # Path to CA bundle
    )

    # Access the specific database
    mongo_db = client['Hotel_Recommendation']

     # Access the new_ratings collection from the database
    new_ratings_collection = mongo_db['new_ratings']
    new_ratings_cursor = new_ratings_collection.find({}, {"_id": 0, "numeric_id": 1, "offering_id": 1, "mean_rating": 1, "username": 1})

    # Convert MongoDB cursor to a DataFrame
    new_ratings_df = pd.DataFrame(list(new_ratings_cursor))

    # Check if the new ratings DataFrame is empty
    if new_ratings_df.empty:
        print("No new ratings.")
        return

    # Insert new ratings into the review_with_index collection
    review_with_index_collection = mongo_db['review_with_index']
    review_with_index_collection.insert_many(new_ratings_df.to_dict('records'))

    cursor = review_with_index_collection.find({}, {"_id": 0, "numeric_id": 1, "offering_id": 1, "mean_rating": 1})

    # Convert MongoDB cursor to a DataFrame
    df = pd.DataFrame(list(cursor))
    df['numeric_id'] = df['numeric_id'].astype(int)

    schema = StructType([
        StructField("offering_id", IntegerType(), True),
        StructField("mean_rating", FloatType(), True),
        StructField("numeric_id", IntegerType(), True)
    ])

    spark_df = spark.createDataFrame(df,  schema=schema)

    # Calculate weights
    weighted_df = calculate_weights(spark_df)
    
    # Initialize ALS model
    als = ALS(maxIter=10, regParam=0.1, userCol="numeric_id", itemCol="offering_id", ratingCol="mean_rating", coldStartStrategy="drop")
    
    # Define the evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="mean_rating", predictionCol="prediction")

    # Split the data into training and testing sets
    (training_data, test_data) = weighted_df.randomSplit([0.8, 0.2])
    
    # Fit the ALS model
    model = als.fit(training_data)

    # Make predictions on the test data
    predictions = model.transform(test_data)

    # Evaluate the model
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse}")

    # Save the model
    model_path = "/Users/yaelhassid/final_project3.0/models/als_model"
    model.write().overwrite().save(model_path)
    print("Model updated and saved successfully.")

    # Delete all documents in the collection
    result = new_ratings_collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the collection.")


# Bash command to copy the model to the Kubernetes pod
copy_model_command = """
kubectl cp /Users/yaelhassid/final_project3.0/models/als_model staysmart-app-deployment-75b99c5c7c-xqsmh:models
"""

with DAG(
    'mongo_data_processing_dag',
    default_args=default_args,
    description='A DAG to extract data from MongoDB and train the model',
    schedule='@daily',  # Change from schedule_interval to schedule
    catchup=False,
) as dag:
    
    extract_and_train = PythonOperator(
    task_id='extract_data_and_train',
    python_callable=extract_and_train,
    )

    # Define a BashOperator to run the command
    copy_model_task = BashOperator(
    task_id='copy_model_to_pod',
    bash_command=copy_model_command,
    dag=dag,
    )

extract_and_train>>copy_model_task
