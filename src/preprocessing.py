import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load parameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

def preprocess_data():
    # Load data based on params
    try:
        cars = pd.read_csv(params['preprocessing']['input_data'])
    except FileNotFoundError:
        print(f"Error: {params['preprocessing']['input_data']} not found")
        return

    # Handle categorical variables
    cars["Spare key"] = cars["Spare key"].map({"Yes": 1, "No": 0})
    cars["Transmission"] = cars["Transmission"].map({"Manual": 1, "Automatic": 0})

    # Handle missing values in 'Fuel type' and encode 'Fuel type' using OneHotEncoder
    cars['Fuel type'] = cars['Fuel type'].fillna('Unknown')

    encoder = OneHotEncoder(drop='first', sparse=False)
    fuel_encoded = encoder.fit_transform(cars[['Fuel type']])
    fuel_encoded_df = pd.DataFrame(fuel_encoded, columns=encoder.get_feature_names_out(['Fuel type']))

    # Drop original 'Fuel type' column and concatenate the encoded features
    cars = cars.drop(columns=['Fuel type'])
    cars = pd.concat([cars, fuel_encoded_df], axis=1)

    # Split the data into X (features) and y (target)
    X = cars.drop(columns=['Price', 'Model Name'])
    y = cars['Price']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add target variable back for later use in feature creation
    X_train['Price'] = y_train
    X_test['Price'] = y_test

    # Ensure output directories exist before saving the files
    os.makedirs(os.path.dirname(params['preprocessing']['output_train_data']), exist_ok=True)
    os.makedirs(os.path.dirname(params['preprocessing']['output_test_data']), exist_ok=True)

    # Save processed data based on params
    X_train.to_csv(params['preprocessing']['output_train_data'], index=False)
    X_test.to_csv(params['preprocessing']['output_test_data'], index=False)
    
    print(f"Preprocessing complete. Files saved to {params['preprocessing']['output_train_data']} and {params['preprocessing']['output_test_data']}")

if __name__ == "__main__":
    preprocess_data()