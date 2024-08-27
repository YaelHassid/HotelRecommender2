import random
from flask import Flask, json, request, render_template, redirect, url_for, flash, session, jsonify
import os
import certifi
from pymongo import MongoClient
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField, IntegerField, TextAreaField, FloatField
from wtforms.validators import DataRequired, Length, EqualTo
from wtforms.widgets import TextInput
from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALSModel
from flask.cli import load_dotenv
import logging
from bson import ObjectId
from confluent_kafka import Producer
import json
import sys
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize CSRF protection
csrf = CSRFProtect(app)

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

# Access the database and collection
db = client[mongo_db]
users_collection = db["users"]
reviews_collection = db["review_with_index"]
offerings_collection = db["offerings"]
google_places_collection = db["google_places"]

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class AutocompleteInput(TextInput):
    def __call__(self, field, **kwargs):
        kwargs.setdefault('data-provide', 'typeahead')
        kwargs.setdefault('autocomplete', 'off')
        return super().__call__(field, **kwargs)

class RatingForm(FlaskForm):
    hotel_name = StringField('Hotel Name', validators=[DataRequired()], widget=AutocompleteInput())
    offering_id = IntegerField('Hotel ID', validators=[DataRequired()])
    rating = FloatField('Rating', validators=[DataRequired()])
    review = TextAreaField('Review', validators=[Length(max=500)])
    submit = SubmitField('Submit Rating')

@app.route('/autocomplete_hotel', methods=['GET'])
def autocomplete_hotel():
    search = request.args.get('q')
    if search:
        hotels = offerings_collection.find({"name": {"$regex": search, "$options": "i"}})
        results = [{"id": str(hotel["id"]), "name": hotel["name"], "address": hotel.get("address", "")} for hotel in hotels]
        return jsonify(matching_results=results)
    return jsonify(matching_results=[])


# Kafka producer configuration
kafka_config = {
    'bootstrap.servers': 'kafka:9092'  # Replace with your Kafka server
}
producer = Producer(kafka_config)

@app.route('/rate_hotel', methods=['GET', 'POST'])
@login_required
def rate_hotel():
    form = RatingForm()
    if form.validate_on_submit():
        offering_id = form.offering_id.data
        rating = form.rating.data
        review = form.review.data
        user_id = current_user.numeric_id

        # Ensure numeric_id is an integer
        if not isinstance(user_id, int):
            user_id = int(user_id)

        # Produce message to Kafka
        message = {
            'user_id': user_id,
            'offering_id': offering_id,
            'mean_rating': rating,
            'review': review,
            'username': current_user.username
        }
        producer.produce('hotel_ratings', key=str(user_id), value=json.dumps(message))
        producer.flush()

        flash('Your rating has been submitted!', 'success')
        return redirect(url_for('home'))
    
    return render_template('rate_hotel.html', form=form)

def generate_unique_numeric_id():
    while True:
        # Generate a random integer in the range of 0 to 2147483647 (32-bit signed integer)
        numeric_id = random.randint(0, 2147483647)
        if not (users_collection.find_one({"numeric_id": numeric_id}) or reviews_collection.find_one({"numeric_id": numeric_id})):
            return numeric_id

class User(UserMixin):
    def __init__(self, id, username, password_hash, numeric_id=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.numeric_id = numeric_id

    @staticmethod
    def get(numeric_id):
        logging.info(f"Attempting to retrieve user with numeric ID: {numeric_id}")
        user_data = users_collection.find_one({"numeric_id": numeric_id})
        if user_data:
            return User(str(user_data["_id"]), user_data["username"], user_data["password_hash"], user_data.get("numeric_id"))
        return None

    @staticmethod
    def create(username, password_hash):
        user_data = reviews_collection.find_one({"username": username})
        if user_data:
            numeric_id = user_data["numeric_id"]
        else:
            numeric_id = generate_unique_numeric_id()
        user_id = users_collection.insert_one({
            "username": username,
            "password_hash": password_hash,
            "numeric_id": numeric_id
        }).inserted_id
        return User(str(user_id), username, password_hash, numeric_id)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()  
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if users_collection.find_one({"username": username}):
            flash('Username already exists')
            return redirect(url_for('signup'))

        new_user = User.create(username, hashed_password)
        login_user(new_user)
        flash('Your account has been created!', 'success')
        return redirect(url_for('home'))

    return render_template('signup.html', form=form)

@login_manager.user_loader
def load_user(user_id):
    logging.info(f'Loading user with ID: {user_id}')
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        user = User.get(user_data["numeric_id"])
        if user:
            logging.info(f'User {user.username} loaded successfully')
            return user
    logging.info(f'User with ID {user_id} not found')
    return None

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Recommendation System") \
    .getOrCreate()


@app.route('/')
@login_required
def home():
    # Load the ALSModel directly
    recommender_model = ALSModel.load('models/als_model')
    logging.info('Entered home route')
    logging.info(f'Session data: {session}')
    logging.info(f'Current user: {current_user}, Authenticated: {current_user.is_authenticated}')
    if current_user.is_authenticated:
        logging.info(f'Current user: {current_user.username}')
        recommendations = get_user_recommendations(current_user.username, recommender_model)
        return render_template('home.html', username=current_user.username, recommendations=recommendations)
    else:
        logging.info('No user is currently authenticated')
        return redirect(url_for('login'))
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user_data = users_collection.find_one({"username": username})

        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User.get(user_data['numeric_id'])
            if not user.numeric_id:
                user.numeric_id = generate_unique_numeric_id()
                users_collection.update_one({"_id": user_data["_id"]}, {"$set": {"numeric_id": user.numeric_id}})
            login_user(user)
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))

        flash('Login unsuccessful. Please check username and password', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html', form=form)


def get_hotel_details(hotel_name, postal_code):
    API_KEY = os.getenv('API_KEY')
    query = f"{hotel_name}"
    search_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={API_KEY}"
    response = requests.get(search_url)
    search_results = response.json()

    if search_results['status'] == 'OK':
        filtered_results = [place for place in search_results['results'] if postal_code in place['formatted_address']]
        if filtered_results:
            place_id = filtered_results[0]['place_id']
            details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={API_KEY}"
            details_response = requests.get(details_url)
            hotel_details = details_response.json()

            if hotel_details['status'] == 'OK':
                hotel_info = hotel_details['result']
                
                # Extract specific details
                name = hotel_info.get('name', 'N/A')
                address = hotel_info.get('formatted_address', 'N/A')
                phone_number = hotel_info.get('formatted_phone_number', 'N/A')
                website = hotel_info.get('website', 'N/A')
                open_now = hotel_info.get('opening_hours', {}).get('open_now', 'N/A')

                # Extract photo URL if available
                photo_url = None
                if 'photos' in hotel_info and len(hotel_info['photos']) > 0:
                    photo_reference = hotel_info['photos'][0]['photo_reference']
                    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={API_KEY}"
                
                # Create a dictionary to return the desired information
                result = {
                    'name': name,
                    'address': address,
                    'phone_number': phone_number,
                    'website': website,
                    'open_now': open_now,
                    'photo_url': photo_url
                }
                
                return result
            else:
                return None

        else:
            return None
    else:
        return None

def get_hotel_name_postal_code(offering_id):    
    # Find the hotel document in the offerings collection using the offering_id
    hotel = offerings_collection.find_one({"id": offering_id})
    
    if hotel:
        # Get the hotel name
        hotel_name = hotel.get("name", "Unknown")

        # Get the address information
        address = hotel.get("address", {})

        # Check if address is a string and try to clean and parse it
        if isinstance(address, str):
            try:
                # Clean up the string to ensure proper JSON format
                cleaned_address = address.strip()  # Remove leading/trailing whitespace

                # If necessary, handle specific known issues here
                # For example, replace single quotes with double quotes if needed
                cleaned_address = cleaned_address.replace("'", '"')

                # Try parsing the cleaned string as JSON
                address = json.loads(cleaned_address)

            except json.JSONDecodeError:
                logging.error(f"Failed to parse address JSON. Cleaned address: {cleaned_address}")
                address = {}  # Default to empty dictionary if parsing fails

        # Check if address is a dictionary
        if isinstance(address, dict):
            postal_code = address.get('postal-code', 'Not available')
        else:
            # Log the type of address and its content if it's not a dictionary
            logging.error(f"Address is not in expected dictionary format. Type: {type(address)}. Content: {address}")
            postal_code = 'Not available'

        return hotel_name, postal_code
    else:
        logging.warning(f"Hotel with offering_id '{offering_id}' not found.")
        return "Hotel not found", "Not available"

def get_user_recommendations(username, recommender_model):
    user_reviews = reviews_collection.find_one({"username": username})
    if user_reviews:
        logging.info("Found the user")
        numeric_id = user_reviews["numeric_id"]
        
        # Create a DataFrame with the user's numeric ID
        customer_df = spark.createDataFrame([Row(numeric_id=int(numeric_id))])
        logging.info("Created a DataFrame")
        
        # Generate recommendations for the user
        recommendations = recommender_model.recommendForUserSubset(customer_df, 20)
        logging.info("Finished finding recommendations")
        
        # Convert recommendations to Pandas DataFrame
        recommendations_pd = recommendations.toPandas()
        logging.info(f"Recommendations DataFrame: {recommendations_pd}")
        
        # Check if there are any recommendations for the user
        if not recommendations_pd.empty and 'recommendations' in recommendations_pd.columns:
            recommended_items = recommendations_pd['recommendations'][0]
        else:
            logging.info(f"No recommendations found for user: {username}, returning 20 most popular")
            recommendations = recommender_model.recommendForAllUsers(20).toPandas()
            recommended_items = []
            for rec in recommendations['recommendations']:
                recommended_items.extend(rec)
    else:
        # If user does not have reviews, return top 10 recommended hotels
        recommendations = recommender_model.recommendForAllUsers(10).toPandas()
        recommended_items = []
        for rec in recommendations['recommendations']:
            recommended_items.extend(rec)
        recommended_items = recommended_items[:10]

    # Fetch details for each recommended hotel
    hotels_with_details = []
    for item in recommended_items:
        # offering_id = item['offering_id']  # Ensure this key exists in your item
        # hotel_name, postal_code = get_hotel_name_postal_code(offering_id)
        # hotel_details = get_hotel_details(hotel_name, postal_code)
        
        # if hotel_details:
        #     hotels_with_details.append(hotel_details)
        hotel_details = google_places_collection.find_one({'offering_id': item['offering_id']})
        if hotel_details:
            hotels_with_details.append(hotel_details)

    return hotels_with_details


@app.route('/logout')
@login_required
def logout():
    logging.info('Logging out user')
    logout_user()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5050)
