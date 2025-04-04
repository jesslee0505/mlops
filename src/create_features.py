import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
import yaml
import pickle

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["features"]
chi2percentile = params["chi2percentile"]
train_path = params["train_path"]
test_path = params["test_path"]

# Load the data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Feature selection function
def select_features(X_train, y_train, percentile):
    selector = SelectPercentile(chi2, percentile=percentile)
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected, selector

# Feature engineering and saving
def create_features():
    # Load train and test data
    train_data, test_data = load_data(train_path, test_path)
    
    # Separate features and target for train and test data
    X_train = train_data.drop('target', axis=1)  # Assuming the target column is named 'target'
    y_train = train_data['target']
    
    # Select features using the chi2 test
    X_train_selected, selector = select_features(X_train, y_train, chi2percentile)
    
    # Save the transformed datasets
    train_data_selected = pd.DataFrame(X_train_selected, columns=X_train.columns[selector.get_support()])
    test_data_selected = selector.transform(test_data)  # Apply same transformation to test data
    
    # Save to CSV
    train_data_selected.to_csv("data/processed_train_data.csv", index=False)
    pd.DataFrame(test_data_selected, columns=X_train.columns[selector.get_support()]).to_csv("data/processed_test_data.csv", index=False)
    
    # Save the pipeline for later use
    with open('data/pipeline.pkl', 'wb') as f:
        pickle.dump(selector, f)

if __name__ == "__main__":
    create_features()