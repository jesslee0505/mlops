from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import mlflow

class CarPriceTrainingFlow(FlowSpec):
    """
    Metaflow for training car price prediction models.
    This flow handles:
    1. Data ingestion
    2. Feature transformations
    3. Model training and evaluation
    4. Model registration with MLFlow
    """
    
    # Parameters for the flow
    data_path = Parameter('data_path', 
                         help='Path to the raw data file',
                         default='../data/cars24data.csv')
    
    test_size = Parameter('test_size', 
                         help='Percentage of data to use for testing',
                         default=0.2)
    
    random_state = Parameter('random_state', 
                           help='Random seed for reproducibility',
                           default=42)

    @step
    def start(self):
        """
        Start the flow by loading data
        """
        print("Starting the flow...")
        print(f"Loading data from {self.data_path}")
        
        # Load the data
        try:
            self.cars = pd.read_csv(self.data_path)
            print(f"Loaded data with shape {self.cars.shape}")
            
            # Print column info for debugging
            print("Columns in dataset:")
            print(self.cars.columns.tolist())
            
            # Keep only the required columns if they exist
            required_columns = [
                'Spare key', 
                'Transmission',
                'KM driven',
                'Ownership',
                'Imperfections',
                'Repainted Parts',
                'Fuel type',
                'Price'  # target variable
            ]
            
            # Check which columns actually exist in the dataset
            existing_columns = [col for col in required_columns if col in self.cars.columns]
            
            if len(existing_columns) < len(required_columns):
                missing_columns = [col for col in required_columns if col not in existing_columns]
                print(f"Warning: Missing some required columns: {missing_columns}")
            
            # Keep only the columns we need
            self.cars = self.cars[existing_columns]
            print(f"Using columns: {existing_columns}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        self.next(self.process_features)
        
    @step
    def process_features(self):
        """
        Process and transform features for modeling
        """
        print("Processing features...")
        
        # 1. Handle 'Spare key' - binary categorical (Yes=1, No=0)
        if 'Spare key' in self.cars.columns:
            print("Converting 'Spare key' to numeric")
            self.cars["Spare key"] = self.cars["Spare key"].map({"Yes": 1, "No": 0})
            # Handle other values or missing values
            self.cars["Spare key"] = pd.to_numeric(self.cars["Spare key"], errors='coerce')
            self.cars["Spare key"] = self.cars["Spare key"].fillna(0)
        
        # 2. Handle 'Transmission' - one-hot encode
        if 'Transmission' in self.cars.columns:
            print("One-hot encoding 'Transmission'")
            # Handle missing values
            self.cars['Transmission'] = self.cars['Transmission'].fillna('Unknown')
            
            # One-hot encode
            transmission_encoder = OneHotEncoder(drop='first', sparse_output=False)
            transmission_encoded = transmission_encoder.fit_transform(self.cars[['Transmission']])
            transmission_cols = transmission_encoder.get_feature_names_out(['Transmission'])
            transmission_df = pd.DataFrame(transmission_encoded, 
                                          columns=transmission_cols, 
                                          index=self.cars.index)
            
            # Save the encoder for scoring
            self.transmission_encoder = transmission_encoder
            
            # Drop original column and add encoded features
            self.cars = self.cars.drop(columns=['Transmission'])
            self.cars = pd.concat([self.cars, transmission_df], axis=1)
        
        # 3. Handle 'Fuel type' - one-hot encode
        if 'Fuel type' in self.cars.columns:
            print("One-hot encoding 'Fuel type'")
            # Handle missing values
            self.cars['Fuel type'] = self.cars['Fuel type'].fillna('Unknown')
            
            # One-hot encode
            fuel_encoder = OneHotEncoder(drop='first', sparse_output=False)
            fuel_encoded = fuel_encoder.fit_transform(self.cars[['Fuel type']])
            fuel_cols = fuel_encoder.get_feature_names_out(['Fuel type'])
            fuel_df = pd.DataFrame(fuel_encoded, 
                                  columns=fuel_cols, 
                                  index=self.cars.index)
            
            # Save the encoder for scoring
            self.fuel_encoder = fuel_encoder
            
            # Drop original column and add encoded features
            self.cars = self.cars.drop(columns=['Fuel type'])
            self.cars = pd.concat([self.cars, fuel_df], axis=1)
        
        # 4. Ensure all numeric columns are properly typed
        numeric_columns = ['KM driven', 'Ownership', 'Imperfections', 'Repainted Parts', 'Price']
        for col in numeric_columns:
            if col in self.cars.columns:
                print(f"Ensuring {col} is numeric")
                self.cars[col] = pd.to_numeric(self.cars[col], errors='coerce')
        
        # 5. Drop any rows with missing values
        before_shape = self.cars.shape
        self.cars = self.cars.dropna()
        after_shape = self.cars.shape
        print(f"Dropped {before_shape[0] - after_shape[0]} rows with missing values")
        
        print("Feature processing complete. Final columns:")
        print(self.cars.columns.tolist())
        
        self.next(self.prepare_train_test)
        
    @step
    def prepare_train_test(self):
        """
        Prepare train and test sets
        """
        print("Preparing train and test sets...")
        
        # Defining target column
        self.target_column = 'Price'
        
        # Define feature columns (all except target)
        feature_columns = [col for col in self.cars.columns if col != self.target_column]
        
        print(f"Feature columns: {feature_columns}")
        print(f"Target column: {self.target_column}")
        
        # Split features and target
        X = self.cars[feature_columns]
        y = self.cars[self.target_column]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Convert to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.y_train = self.y_train.to_numpy()
        self.y_test = self.y_test.to_numpy()
        
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        self.feature_columns = feature_columns
        
        self.next(self.train_model)
        
    @step
    def train_model(self):
        """
        Train a model on the data
        """
        print("Training model...")
        
        # Set up MLFlow - create experiment if it doesn't exist
        import mlflow
        from mlflow.exceptions import MlflowException
        
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        experiment_name = 'metaflow-experiment'
        
        # Get or create the experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Error with MLflow experiment setup: {e}")
            print("Creating a new experiment...")
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except Exception as e2:
                print(f"Failed to create experiment: {e2}")
                print("Will continue without MLflow logging")
                experiment_id = None
        
        print(f"Using experiment ID: {experiment_id}")
        if experiment_id is not None:
            mlflow.set_experiment(experiment_name)
        
        # Create and train a random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        train_pred = model.predict(self.X_train)
        train_mae = np.mean(np.abs(self.y_train - train_pred))
        
        test_pred = model.predict(self.X_test)
        test_mae = np.mean(np.abs(self.y_test - test_pred))
        
        print(f"Train MAE: {train_mae}")
        print(f"Test MAE: {test_mae}")
        
        # Store the model and metrics
        self.model = model
        self.train_mae = train_mae
        self.test_mae = test_mae
        
        # Log in MLFlow
        if experiment_id is not None:
            try:
                with mlflow.start_run(run_name="car_price_model"):
                    mlflow.log_param("model_type", "RandomForestRegressor")
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_metric("train_mae", train_mae)
                    mlflow.log_metric("test_mae", test_mae)
                    
                    # Log feature importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        for i, importance in enumerate(importances):
                            feature_name = self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}"
                            mlflow.log_metric(f"importance_{feature_name}", importance)
                        
                        # Print feature importances
                        print("Feature importances:")
                        for i, imp in enumerate(importances):
                            feature_name = self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}"
                            print(f"{feature_name}: {imp:.4f}")
            except Exception as e:
                print(f"Error during MLflow logging: {e}")
                print("Continuing without MLflow logging")
        
        self.next(self.register_model)
        
    @step
    def register_model(self):
        """
        Register the model with MLFlow
        """
        print("Registering the model with MLFlow...")
        
        # Import needed libraries
        import mlflow
        from mlflow.exceptions import MlflowException
        
        # Setup MLFlow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        experiment_name = 'metaflow-experiment'
        
        # Get or create the experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error with MLflow experiment setup: {e}")
            print("Will continue without MLflow model registration")
            experiment_id = None
        
        # Register the model in MLFlow
        if experiment_id is not None:
            try:
                with mlflow.start_run(run_name="model_registration"):
                    # Log metrics
                    mlflow.log_metric("train_mae", self.train_mae)
                    mlflow.log_metric("test_mae", self.test_mae)
                    
                    # Log the feature columns for later use in scoring
                    mlflow.log_param("feature_columns", str(self.feature_columns))
                    
                    # Register the model
                    mlflow.sklearn.log_model(
                        self.model,
                        artifact_path="car_price_model",
                        registered_model_name="car-price-predictor"
                    )
                    
                    # Log encoders as pickle artifacts if they exist
                    if hasattr(self, 'transmission_encoder'):
                        mlflow.sklearn.log_model(
                            self.transmission_encoder,
                            artifact_path="transmission_encoder"
                        )
                    
                    if hasattr(self, 'fuel_encoder'):
                        mlflow.sklearn.log_model(
                            self.fuel_encoder,
                            artifact_path="fuel_encoder"
                        )
                    
                    print(f"Model registered in MLFlow with test MAE: {self.test_mae}")
            except Exception as e:
                print(f"Error during MLflow model registration: {e}")
                print("Continuing without MLflow model registration")
        else:
            print("Skipping MLflow model registration due to previous setup errors")
        
        # Create a simple model file as a backup in case MLFlow fails
        import pickle
        try:
            with open('car_price_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print("Model saved to car_price_model.pkl as a backup")
        except Exception as e:
            print(f"Error saving model pickle file: {e}")
        
        self.next(self.end)
        
    @step
    def end(self):
        """
        End the flow
        """
        print("Flow completed successfully!")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Training MAE: {self.train_mae}")
        print(f"Test MAE: {self.test_mae}")
        
        # Print feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            print("\nFeature importances:")
            for i, imp in enumerate(importances):
                feature_name = self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}"
                print(f"{feature_name}: {imp:.4f}")

if __name__ == "__main__":
    CarPriceTrainingFlow()