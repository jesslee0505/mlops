import pandas as pd
import numpy as np 
import pickle
import yaml
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

def load_data(train_path, test_path):
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def process_data(train_data, test_data, chi2percentile):
    # Assuming 'Price' is our target variable
    train_y, test_y = train_data['Price'], test_data['Price']

    # Reshape for pipeline
    train_y = train_y.values.reshape((-1,1))
    test_y = test_y.values.reshape((-1,1))

    # Drop target variable 
    train_data = train_data.drop(columns=['Price'])
    test_data = test_data.drop(columns=['Price'])

    # Create pipeline for imputing and scaling numeric variables
    # one-hot encoding categorical variables, and select features based on chi-squared value
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=chi2percentile)),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder='passthrough'
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    # Create new train and test data using the pipeline
    clf.fit(train_data, train_y.ravel())
    train_new = clf.transform(train_data)
    test_new = clf.transform(test_data)

    # Transform to dataframe
    if hasattr(train_new, "toarray"):
        train_new = pd.DataFrame(train_new.toarray())
        test_new = pd.DataFrame(test_new.toarray())
    else:
        train_new = pd.DataFrame(train_new)
        test_new = pd.DataFrame(test_new)
        
    # Add target back
    train_new['Price'] = train_y
    test_new['Price'] = test_y
    
    return train_new, test_new, clf

def save_data(train_new, test_new, train_output, test_output, clf, clf_output):
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    os.makedirs(os.path.dirname(clf_output), exist_ok=True)
    
    # Save processed data
    train_new.to_csv(train_output, index=False)
    test_new.to_csv(test_output, index=False)
    
    # Save pipeline
    with open(clf_output, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Feature creation complete. Files saved to {train_output}, {test_output}, and {clf_output}")

if __name__=="__main__":
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    features_params = params["features"]
    create_features_params = params["create_features"]
    
    chi2percentile = features_params["chi2percentile"]
    train_path = create_features_params["input_train_data"]
    test_path = create_features_params["input_test_data"]
    train_output = create_features_params["output_train_data"]
    test_output = create_features_params["output_test_data"]
    pipeline_output = create_features_params["pipeline_output"]

    train_data, test_data = load_data(train_path, test_path)
    train_new, test_new, clf = process_data(train_data, test_data, chi2percentile)
    save_data(train_new, test_new, train_output, test_output, clf, pipeline_output)