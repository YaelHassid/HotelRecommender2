import json
import os
import certifi
from dotenv import load_dotenv  # Add this import
from confluent_kafka import Consumer, KafkaException, KafkaError
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Kafka consumer configuration
kafka_config = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'hotel_recommendation_group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(kafka_config)
consumer.subscribe(['hotel_ratings'])


# MongoDB connection settings
mongo_uri = os.getenv('MONGO_URI')
mongo_db = "Hotel_Recommendation"

# Use certifi to get the path to the CA bundle
ca = certifi.where()

# Initialize the MongoClient with TLS settings
client = MongoClient(
    mongo_uri,
    tls=True,  # Enable TLS
    tlsCAFile=ca  # Path to CA bundle
)

def insert_new_rating(data):
    # Access the database and collection
    db = client[mongo_db]
    new_ratings_collection = db["new_ratings"]
    try:
        # Insert the new rating data
        new_ratings_collection.insert_one(data)
        print("New rating inserted successfully.")
    except Exception as e:
        print(f"Error inserting new rating: {e}")

def consume_messages():
    print("Entered the consume_messages function")
    while True:
        try:
            msg = consumer.poll(1.0)  # Timeout of 1 second

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print("End of partition reached {0}/{1}".format(msg.topic(), msg.partition()))
                    continue
                else:
                    print(f"Kafka error: {msg.error()}")
                    break
            message = json.loads(msg.value().decode('utf-8'))
            print(f"Consumed message: {message}")

            # Convert message to the required schema format
            new_data = {
                "numeric_id": int(message["user_id"]),
                "offering_id": int(message["offering_id"]),
                "mean_rating": float(message["mean_rating"]),
                "username": str(message["username"])
            }
            insert_new_rating(new_data)
        except KafkaException as e:
            print(f"KafkaException: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    consumer.close()

if __name__ == "__main__":
    try:
        consume_messages()
    except KeyboardInterrupt:
        print("Aborted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
