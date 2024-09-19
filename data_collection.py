import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
print("scikit-learn is working correctly.")
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Step 1: Extract - Scrape movie data
def scrape_movie_data():
    url = "https://www.imdb.com/chart/top/"
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    movies = []
    for item in soup.find_all('div', class_='sc-b189961a-0 iqHBGn cli-children'):
        movie_title = item.find('h3', class_='ipc-title__text').text.strip()
        
        # Extract the link from the a tag
        link_tag = item.find('a', class_='ipc-title-link-wrapper')
        if link_tag and 'href' in link_tag.attrs:
            link = 'https://www.imdb.com' + link_tag['href']
        else:
            link = "N/A"

        metadata_div = item.find('div', class_='sc-b189961a-7 btCcOY cli-title-metadata')
        year_span = metadata_div.find('span', class_='sc-b189961a-8 hCbzGp cli-title-metadata-item') if metadata_div else None
        year = year_span.text.strip() if year_span else "N/A"
        movies.append([movie_title, year, link])

    return pd.DataFrame(movies, columns=['Title', 'Year', 'Link'])

# Step 2: Transform - Clean and preprocess data
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing data (drop rows with missing values in 'Year')
    df = df[df['Year'] != 'N/A']
    df['Year'] = df['Year'].astype(int)

    # Normalize data (if applicable - in this case, we'll just ensure Year is numeric)
    df.dropna(inplace=True)
    
    return df

# Step 3: Load - Store cleaned data into Firebase Firestore
def store_data_to_firestore(df):
    collection_ref = db.collection('movies')

    # Clear existing data (optional, to avoid duplicates)
    docs = collection_ref.stream()
    for doc in docs:
        doc.reference.delete()

    # Insert new data
    for _, row in df.iterrows():
        doc_ref = collection_ref.add({
            'title': row['Title'],
            'year': row['Year'],
            'link': row['Link']
        })

# Step 4: Machine Learning - Train Random Forest model to predict movie ratings
def train_model(df):
    # Using 'Year' as a feature (replace with more features as needed)
    X = df[['Year']].values
    y = np.random.randint(1, 10, len(df))  # Generate random movie ratings (for illustration)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# Step 5: Build an API to expose model predictions via Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from POST request
    data = request.get_json()

    # Extract features (Year in this case)
    year = data.get('Year')
    features = np.array([[year]])

    # Make prediction using trained model
    prediction = model.predict(features)[0]
    
    return jsonify({'predicted_rating': prediction})

if __name__ == '__main__':
    # Run ETL Pipeline
    movies_df = scrape_movie_data()
    print("Scraped Movie Data:")
    print(movies_df.head())

    cleaned_df = clean_data(movies_df)
    print("Cleaned Movie Data:")
    print(cleaned_df.head())

    store_data_to_firestore(cleaned_df)

    # Train the model and make it available for prediction
    model = train_model(cleaned_df)

    # Run Flask app
    app.run(debug=True)