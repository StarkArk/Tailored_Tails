# final_final.py
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import csv

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("breed_rf_model.pkl")

# Load breed mapping from CSV file
breed_mapping = {}
breed_info = {}
with open('encoded_labels.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        breed_mapping[int(row[0])] = row[1]
        breed_info[row[1]] = {'description': row[2], 'image_url': row[3]}

# Define the home route
@app.route('/')
def index():
    return render_template('final_project.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Map dropdown values to numerical representation
    feature_mapping = {
        'Low': 0, 'Moderate': 1, 'High': 2,
        'Short': 0, 'Moderate': 1, 'Long': 2,
        'Bronx': 0, 'Brooklyn': 1, 'Manhattan': 2, 'Queens': 3, 'Staten Island': 4,
        'Yes': 1, 'No': 0,
        'High': 2, 'Low': 0, 'Middle': 1,
        'Large (50lb+)': 2, 'Medium (36-49lb)': 1, 'Small (9-35lb)': 0,
        'High': 2, 'Low': 0, 'Medium': 1
    }
    
    # Extract features from the input data
    features = [
        'grooming_frequency', 'shedding', 'energy_level',
        'trainability', 'demeanor', 'life_expectancy',
        'borough', 'dog_friendly', 'income',
        'dog_size', 'lifetime_cost'
    ]
    
    # Convert dropdown values to numerical representation
    input_features = [feature_mapping[data[feature]] for feature in features]
    
    # Ensure all 24 features are present
    input_features.extend([0] * (24 - len(input_features)))
    
    # Convert input data to numpy array
    input_features_array = np.array(input_features).reshape(1, -1)
    
    # Make predictions using the model
    prediction = model.predict(input_features_array)
    
    # Convert numerical prediction to breed name using the mapping dictionary
    predicted_breed = breed_mapping[prediction[0]]
    
    # Get breed description and image URL from breed_info
    breed_data = breed_info.get(predicted_breed, {'description': 'Description not found', 'image_url': ''})
    
    # Return the predicted breed, description, and image URL as JSON response
    return jsonify({'prediction': predicted_breed, 'description': breed_data['description'], 'image_url': breed_data['image_url']})


# Define the breed info route
@app.route('/breed_info/<breed_name>')
def get_breed_info(breed_name):
    # Retrieve breed information based on breed name
    breed_data = breed_info.get(breed_name)
    if breed_data:
        return jsonify(breed_data)
    else:
        return jsonify({'description': 'Description not found', 'image_url': ''})


if __name__ == '__main__':
    app.run(debug=True)
