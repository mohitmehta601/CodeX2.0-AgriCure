#!/usr/bin/env python3
"""
Script to retrain the fertilizer recommendation model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

def retrain_model():
    """Retrain the model with current scikit-learn version"""
    try:
        # Load the dataset
        data_path = "../Ml-model-main/f2.csv"
        if not os.path.exists(data_path):
            data_path = "Ml-model-main/f2.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError("Dataset f2.csv not found")
        
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare features and target
        X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
        y = df['Fertilizer']
        
        # Create label encoders for categorical variables
        soil_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()
        fertilizer_encoder = LabelEncoder()
        
        # Encode categorical variables
        X_encoded = X.copy()
        X_encoded['Soil_Type'] = soil_encoder.fit_transform(X['Soil_Type'])
        X_encoded['Crop_Type'] = crop_encoder.fit_transform(X['Crop_Type'])
        y_encoded = fertilizer_encoder.fit_transform(y)
        
        print(f"Unique soil types: {soil_encoder.classes_}")
        print(f"Unique crop types: {crop_encoder.classes_}")
        print(f"Unique fertilizers: {fertilizer_encoder.classes_}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model (same parameters as original)
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        print("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        
        # Save the retrained model and encoders
        os.makedirs("models", exist_ok=True)
        
        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("models/fertilizer.pkl", "wb") as f:
            pickle.dump(fertilizer_encoder, f)
        
        # Also save soil and crop encoders for consistency
        with open("models/soil_encoder.pkl", "wb") as f:
            pickle.dump(soil_encoder, f)
            
        with open("models/crop_encoder.pkl", "wb") as f:
            pickle.dump(crop_encoder, f)
        
        print("Model and encoder saved successfully")
        
        # Test prediction
        test_input = np.array([[25, 78, 43, 4, 1, 22, 26, 38]])
        prediction = model.predict(test_input)
        fertilizer = fertilizer_encoder.classes_[prediction[0]]
        print(f"Test prediction: {fertilizer}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Retraining fertilizer recommendation model...")
    success = retrain_model()
    if success:
        print("\n🎉 Model retrained successfully!")
    else:
        print("\n💥 Model retraining failed!")