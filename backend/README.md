# AgriCure ML API Backend

This FastAPI backend provides machine learning-powered fertilizer recommendations using a Random Forest classifier trained on agricultural data.

## Features

- **Machine Learning Model**: Random Forest classifier trained on soil and crop data
- **Real-time Predictions**: Fast API endpoints for fertilizer recommendations
- **Data Validation**: Input validation and error handling
- **CORS Support**: Cross-origin requests enabled for frontend integration
- **Health Monitoring**: Health check and model status endpoints

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check with model status

### Model Information
- `GET /model-info` - Get model details, accuracy, and available options

### Predictions
- `POST /predict` - Get fertilizer recommendation

#### Prediction Input Format:
```json
{
  "Temperature": 25.0,
  "Humidity": 70.0,
  "Moisture": 60.0,
  "Soil_Type": "Loamy",
  "Crop_Type": "Wheat",
  "Nitrogen": 45.0,
  "Potassium": 150.0,
  "Phosphorous": 25.0
}
```

#### Prediction Output Format:
```json
{
  "fertilizer": "Urea",
  "confidence": 92.5,
  "model_accuracy": 96.8
}
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python run_local.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Model Training

The model is automatically trained on startup using the dataset from `../Ml-model-main/f2.csv`. The trained model and encoders are saved to the `models/` directory for faster subsequent startups.

## Deployment

The backend can be deployed using Docker:

```bash
docker build -t agricure-ml-api .
docker run -p 8000:8000 agricure-ml-api
```

## Environment Variables

- `ML_API_URL`: Base URL for the ML API (used by frontend)
- `PORT`: Port to run the server on (default: 8000)