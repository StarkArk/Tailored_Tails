# Import the required modules
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
     
# Read the preprocessed_doggy file into a DataFrame
doggy_df = pd.read_csv(
    Path("preprocessed_doggy.csv")
)

# Review the DataFrame
display(doggy_df.head())
display(doggy_df.tail())


# Remove the Unnamed col
doggy_df = doggy_df[['Borough', 'dog_friendly', 'income_cat',
       'grooming_frequency', 'shedding', 'energy_level', 'trainability',
       'demeanor', 'size', 'life_expectancy', 'lifetime_cost', 'Breed']]

doggy_df.info()


# Split the data into features(X) and targets(y)
# First y
y = doggy_df['Breed']

# Next X
X = doggy_df.drop(columns=['Breed'])
     
# Use LabelEncoder to convert targets(50) into a single column
# Use LabelEncoder on the y DataFrame to encode the Dog Breeds
label_encoder_df = LabelEncoder()

# Encode Labels
y_encoded = label_encoder_df.fit_transform(y)

# Test the results of the encoder
# Display first 50 of encoded column
display(y_encoded[0:50])

# Get the original labels back
display(label_encoder_df.inverse_transform(y_encoded[0:50]))

# Convert Features/Targets into a form suitable for Modeling
y = y_encoded

X = pd.get_dummies(X, dtype=int)


# Look at X, y data
display(y[0:10])
display(X.head(3))

# Features
X.shape
     
# Target
y.shape


# Split into testing and training sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {'n_estimators': [50, 100, 200, 300]}

# Perform a grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameter and corresponding accuracy
print("Best n_estimators:", grid_search.best_params_['n_estimators'])
print("Best Accuracy:", grid_search.best_score_)
     
# Run the same analysis using results from previous model run to determine best 'n_estimators' param

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {'n_estimators': [20, 35, 50, 70]}

# Perform a grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameter and corresponding accuracy
print("Best n_estimators:", grid_search.best_params_['n_estimators'])
print("Best Accuracy:", grid_search.best_score_)
     
# Run the same analysis using results from previous model run to determine best 'n_estimators' param, 3rd attempt

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {'n_estimators': [10, 15, 20, 25]}

# Perform a grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameter and corresponding accuracy
print("Best n_estimators:", grid_search.best_params_['n_estimators'])
print("Best Accuracy:", grid_search.best_score_)

# Run the same analysis using results from previous model run to determine best 'n_estimators' param, final attempt

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {'n_estimators': [4, 8, 10, 12]}

# Perform a grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameter and corresponding accuracy
print("Best n_estimators:", grid_search.best_params_['n_estimators'])
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the Metrics for model performance
from sklearn.metrics import accuracy_score

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=8, random_state=42)

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Perfect Accuracy on 48 Breeds, 90% and 0% on the remaining 2. What are those Breeds?

# !!!Work on This!!!

print("100% accuracy for 48 Breeds")

print("0% Accuracy in predicting the Breed:")
display(label_encoder_df.inverse_transform([7]))

print("90 Accuracy for predicting the Breed:")
display(label_encoder_df.inverse_transform([40]))

print(np.min(y), np.max(y))

encoded = list(range(50))
labels = label_encoder_df.inverse_transform(encoded)

labels_encoded_df = pd.DataFrame()
labels_encoded_df['encoded'] = encoded
labels_encoded_df['labels'] = labels

display(labels_encoded_df.head(50))

# Export Label DataFrame as csv
labels_encoded_df.to_csv('encoded_labels.csv', index=False)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the DataFrame
print(feature_importance_df)

# Model Attribution by Breed, Zipcode, and Borough info

# Sum up contributions of the Breed Level data
indices_breed = [3, 0, 1, 5, 4, 2, 23, 21, 19, 22, 20, 18, 17, 16]
breed_features_contribution = round(feature_importance_df.loc[indices_breed, 'Importance'].sum() * 100, 2)

# For ZipCode level data
indices_zip = [13, 14, 15, 12, 11]
zip_features_contribution = round(feature_importance_df.loc[indices_zip, 'Importance'].sum() * 100, 2)

# For Borough level data
indices_boro = [8, 9, 6, 7, 10]
borough_features_contribution = round(feature_importance_df.loc[indices_boro, 'Importance'].sum() * 100, 2)

# Print out contributions
print(f'breed level contribution: {breed_features_contribution}%')
print(f'zipcode level contribution: {zip_features_contribution}%')
print(f'Borough level contribution: {borough_features_contribution}%')

# Save the Random Forest model
import joblib

# Save the model to a file
joblib.dump(rf_classifier, 'breed_rf_model.pkl')

loaded_model = joblib.load('breed_rf_model.pkl')

print(loaded_model)


# Display row 100 as a list
test_list = list(X.loc[100])
display(X.loc[100].index)
display(test_list)
     
# Find the possible values in each column
multi_col_list = ['grooming_frequency', 'shedding', 'energy_level', 'trainability','demeanor', 'life_expectancy']

for col in multi_col_list:
  print(sorted(doggy_df[col].unique()), len(doggy_df[col].unique()))
     

# Test Prediction

# Choose traits = > [grooming_frequency, shedding, energy_level, trainability, demeanor, life_expectancy],
# choose [0.2(only available for the first 4), 0.4, 0.6, 0.8, 1.0] for each trait in order
traits = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

# Choose for Borough => ['Borough_Bronx', 'Borough_Brooklyn', 'Borough_Manhattan', 'Borough_Queens', 'Borough_Staten Island'], place a 1 on boro
# and 0 on all other entries
borough = [0, 1, 0, 0, 0]

# Choose Dog Friendly Area = [No, Yes], place a 1 on choice and 0 on all other entries
dog_friendly = [0, 1]

# Choose income area => [High, Low, Middle], place a 1 on choice and 0 on all other entries
income = [0, 0, 1]

# Choose Dog Size => ['Giant <75lb', 'Large 55-75lb', 'Medium 35-55lb', 'Small 9-35lb', 'Toy >9lb'], place a 1 on choice and 0 on all other entries
dog_size = [0, 0, 1, 0, 0]

# Choose Lifetime Cost = ['High', 'Low', 'Medium'], place a 1 on choice and 0 on all other entries
lifetime_cost = [0, 0, 1]

# Combine the lists to create a single input to the model
choices = np.array(traits + borough + dog_friendly + income + dog_size + lifetime_cost)

# Change the shape of the array
choices = choices.reshape(1, -1)

# Model Prediction
predictions = loaded_model.predict(choices)

# Display the predictions
print(predictions)
print(labels_encoded_df['labels'].loc[predictions[0]])

from sklearn.ensemble import RandomForestClassifier
loaded_model.n_features_in_
     
