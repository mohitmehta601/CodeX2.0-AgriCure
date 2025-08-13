import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fertilizer ML API", description="Fertilizer Recommendation ML Model API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and encoders
model = None
fertilizer_encoder = None

class PredictionInput(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float

class PredictionOutput(BaseModel):
    fertilizer: str

def load_existing_model():
    """Load the existing trained model and encoders"""
    global model, fertilizer_encoder
    
    try:
        # Load the newly retrained model
        model_path = "models/classifier.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model models/classifier.pkl not found")
        
        # Load the fertilizer encoder
        encoder_path = "models/fertilizer.pkl"
        if not os.path.exists(encoder_path):
            raise FileNotFoundError("Fertilizer encoder models/fertilizer.pkl not found")
        
        logger.info(f"Loading trained model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loading fertilizer encoder from {encoder_path}")
        with open(encoder_path, 'rb') as f:
            fertilizer_encoder = pickle.load(f)
        
        logger.info("Model and encoder loaded successfully")
        logger.info(f"Available fertilizers: {fertilizer_encoder.classes_}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Starting Fertilizer ML API...")
    load_existing_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Fertilizer ML API is running",
        "model_loaded": model is not None,
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "available_fertilizers": fertilizer_encoder.classes_.tolist() if fertilizer_encoder else []
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_fertilizer(input_data: PredictionInput):
    """Predict fertilizer based on input parameters"""
    
    if model is None or fertilizer_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        # Create a mapping for soil types (string to int)
        soil_type_mapping = {
            "Clayey": 0,
            "Loamy": 1,
            "Red": 2,
            "Black": 3,
            "Sandy": 4
        }
        
        # Create a mapping for crop types (string to int)
        crop_type_mapping = {
            "rice": 0, "Wheat": 1, "Sugarcane": 2, "Pulses": 3, "Paddy": 4,
            "pomegranate": 5, "Oil seeds": 6, "Millets": 7, "Maize": 8,
            "Ground Nuts": 9, "Cotton": 10, "coffee": 11, "watermelon": 12,
            "Barley": 13, "Tobacco": 14, "Jute": 15, "Tea": 16
        }
        
        # Validate soil type
        if input_data.Soil_Type not in soil_type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid Soil_Type. Available options: {list(soil_type_mapping.keys())}"
            )
        
        # Validate crop type
        if input_data.Crop_Type not in crop_type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid Crop_Type. Available options: {list(crop_type_mapping.keys())}"
            )
        
        # Encode categorical variables
        soil_encoded = soil_type_mapping[input_data.Soil_Type]
        crop_encoded = crop_type_mapping[input_data.Crop_Type]
        
        # Prepare input array for prediction (note: using the original column order from the dataset)
        input_array = np.array([[
            input_data.Temperature,  # This will be mapped to 'Temparature' in the model
            input_data.Humidity,
            input_data.Moisture,
            soil_encoded,
            crop_encoded,
            input_data.Nitrogen,
            input_data.Potassium,
            input_data.Phosphorous
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Decode the prediction
        fertilizer = fertilizer_encoder.classes_[prediction[0]]
        
        logger.info(f"Prediction made: {fertilizer}")
        
        return PredictionOutput(fertilizer=fertilizer)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "available_soil_types": ["Clayey", "Loamy", "Red", "Black", "Sandy"],
        "available_crop_types": ["rice", "Wheat", "Sugarcane", "Pulses", "Paddy", "pomegranate", "Oil seeds", "Millets", "Maize", "Ground Nuts", "Cotton", "coffee", "watermelon", "Barley", "Tobacco", "Jute", "Tea"],
        "available_fertilizers": fertilizer_encoder.classes_.tolist() if fertilizer_encoder else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)