import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(train_file, test_file):
    # Read the data
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    # Assuming the target variable is the last column
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Encode categorical features (Assuming categorical features are of type 'object')
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OneHotEncoder for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectPercentile(chi2, percentile=50))
    ])
    
    # Apply transformations
    X_train_selected = pipeline.fit_transform(X_train, y_train)
    X_test_selected = pipeline.transform(X_test)
    
    # Here you can save the transformed data and pipeline
    pd.DataFrame(X_train_selected).to_csv('data/processed_train_data.csv', index=False)
    pd.DataFrame(X_test_selected).to_csv('data/processed_test_data.csv', index=False)
    
    # Save the pipeline
    import joblib
    joblib.dump(pipeline, 'data/pipeline.pkl')

# Example usage
preprocess_data('data/adult.data', 'data/adult.test')
