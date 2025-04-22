from metaflow import FlowSpec, step, Parameter, conda, conda_base, kubernetes, resources, timeout, retry, catch
import pandas as pd
import numpy as np
import mlflow
import pickle

@conda_base(libraries={
    'pandas': '>=1.3.0',
    'numpy': '>=1.20.0',
    'scikit-learn': '>=1.0.0',
    'mlflow': '>=1.20.0'
})
class CarPriceScoringFlowGCP(FlowSpec):
    """
    Metaflow for scoring new data with the car price prediction model.
    """
    
    # Parameters for the flow
    model_name = Parameter('model_name', 
                          help='Name of the registered model in MLFlow',
                          default='car-price-predictor-gcp')
    
    data_path = Parameter('data_path', 
                         help='Path to the new data file for scoring',
                         default='../data/cars24data_new.csv')
    
    output_path = Parameter('output_path',
                          help='Path to save the predictions',
                          default='predictions_gcp.csv')

    @catch(var='load_error')
    @retry(times=3)
    @step
    def start(self):
        """
        Start the flow by loading the model and data
        """
        print("Starting the scoring flow...")
        
        # Check if there was an error from a previous attempt
        if hasattr(self, 'load_error') and self.load_error is not None:
            print(f"Previous attempt had an error: {self.load_error}")
            print("Retrying data loading...")
        
        # Setup MLFlow
        print("Setting up MLFlow connection...")
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Load the model from MLFlow
        try:
            print(f"Loading model: {self.model_name}")
            
            # Get the latest version of the model
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Try to get the model
            try:
                latest_model = client.get_latest_versions(self.model_name, stages=["None"])[0]
                model_uri = f"models:/{self.model_name}/latest"
                print(f"Found model: {latest_model.run_id}")
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Model loaded successfully from MLFlow")
            except Exception as e:
                print(f"Failed to load model from MLFlow: {e}")
                print("Attempting to load model from backup file...")
                try:
                    with open('car_price_model_gcp.pkl', 'rb') as f:
                        self.model = pickle.load(f)
                    print("Model loaded from backup file")
                except Exception as e2:
                    print(f"Failed to load model from backup: {e2}")
                    raise Exception("Could not load model from MLFlow or backup")
            
            # Load run information to get feature columns
            try:
                run = mlflow.get_run(latest_model.run_id)
                self.feature_columns = eval(run.data.params.get('feature_columns', '[]'))
                print(f"Feature columns loaded: {self.feature_columns}")
            except Exception as e:
                print(f"Failed to load feature columns: {e}")
                self.feature_columns = None
                
            # Load encoders if they exist
            try:
                run_id = latest_model.run_id
                client = MlflowClient()
                
                try:
                    transmission_artifact_uri = f"runs:/{run_id}/transmission_encoder"
                    self.transmission_encoder = mlflow.sklearn.load_model(transmission_artifact_uri)
                    print("Transmission encoder loaded")
                except Exception as e:
                    print(f"No transmission encoder found: {e}")
                    self.transmission_encoder = None
                
                try:
                    fuel_artifact_uri = f"runs:/{run_id}/fuel_encoder"
                    self.fuel_encoder = mlflow.sklearn.load_model(fuel_artifact_uri)
                    print("Fuel encoder loaded")
                except Exception as e:
                    print(f"No fuel encoder found: {e}")
                    self.fuel_encoder = None
                    
            except Exception as e:
                print(f"Failed to load encoders: {e}")
                self.transmission_encoder = None
                self.fuel_encoder = None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Load the data
        try:
            print(f"Loading data from {self.data_path}")
            self.new_data = pd.read_csv(self.data_path)
            print(f"Loaded data with shape {self.new_data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
        self.next(self.process_data)
        
    @catch(var='process_error')
    @timeout(minutes=10)
    @retry(times=2)
    @step
    def process_data(self):
        """
        Process the new data for scoring
        """
        print("Processing data for scoring...")
        
        # Create a copy to avoid modifying the original
        self.processed_data = self.new_data.copy()
        
        # Process features...
        # (simplified for brevity)
        
        print(f"Processed data shape: {self.processed_data.shape}")
        self.next(self.generate_predictions)
        
    @catch(var='predict_error')
    @kubernetes(cpu=1, memory=2000)
    @timeout(minutes=10)
    @retry(times=2)
    @step
    def generate_predictions(self):
        """
        Generate predictions using the model
        """
        print("Generating predictions...")
        
        try:
            # Convert data to numpy for prediction
            X = self.processed_data.to_numpy()
            
            # Generate predictions
            self.predictions = self.model.predict(X)
            print(f"Generated {len(self.predictions)} predictions")
            
            # Add predictions to original data
            self.new_data['Predicted_Price'] = self.predictions
            
            # Save the predictions
            self.new_data.to_csv(self.output_path, index=False)
            print(f"Saved predictions to {self.output_path}")
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            raise
            
        self.next(self.end)
        
    @step
    def end(self):
        """
        End the flow
        """
        print("Scoring flow completed successfully!")
        print(f"Results saved to {self.output_path}")
        
        # Show sample of predictions
        sample_size = min(5, len(self.predictions))
        print(f"\nSample of {sample_size} predictions:")
        print(self.new_data.head(sample_size))

if __name__ == "__main__":
    CarPriceScoringFlowGCP()