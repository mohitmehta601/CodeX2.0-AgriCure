#!/usr/bin/env python3
"""
Test script to verify model loading and prediction
"""

import pickle
import numpy as np
import os

def test_model_loading():
    """Test if the trained model can be loaded"""
    try:
        # Try to load the model
        model_path = "../Ml-model-main/classifier.pkl"
        if not os.path.exists(model_path):
            model_path = "Ml-model-main/classifier.pkl"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Try to load the encoder
        encoder_path = "../Ml-model-main/fertilizer.pkl"
        if not os.path.exists(encoder_path):
            encoder_path = "Ml-model-main/fertilizer.pkl"
        
        if not os.path.exists(encoder_path):
            print(f"❌ Encoder file not found at {encoder_path}")
            return False
        
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ Encoder loaded successfully")
        print(f"   Available fertilizers: {encoder.classes_}")
        
        # Test a prediction
        test_input = np.array([[25, 78, 43, 4, 1, 22, 26, 38]])  # Example from the notebook
        prediction = model.predict(test_input)
        fertilizer = encoder.classes_[prediction[0]]
        print(f"✅ Test prediction successful: {fertilizer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing model loading...")
    success = test_model_loading()
    if success:
        print("\n🎉 All tests passed! Model is ready to use.")
    else:
        print("\n💥 Tests failed! Check the model files.")
