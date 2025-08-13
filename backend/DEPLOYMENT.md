# ML API Deployment Guide

## Local Development

The ML API is currently running locally on `http://localhost:8000`.

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python main.py
   ```

3. The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /model-info` - Model information

### Prediction
- `POST /predict` - Get fertilizer recommendation

#### Request Body:
```json
{
  "Temperature": 25.0,
  "Humidity": 70.0,
  "Moisture": 45.0,
  "Soil_Type": "Loamy",
  "Crop_Type": "Wheat",
  "Nitrogen": 45.0,
  "Potassium": 30.0,
  "Phosphorous": 35.0
}
```

#### Response:
```json
{
  "fertilizer": "Urea"
}
```

## Production Deployment

### Option 1: Railway (Recommended for quick deployment)
1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set environment variables if needed
4. Deploy automatically

### Option 2: Render
1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option 3: Heroku
1. Create account at [heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Create app and deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 4: DigitalOcean App Platform
1. Create account at [digitalocean.com](https://digitalocean.com)
2. Go to App Platform
3. Connect your repository
4. Configure as Python app

## Environment Variables

For production, you may want to set:
- `PORT` - Port number (default: 8000)
- `HOST` - Host binding (default: 0.0.0.0)

## CORS Configuration

The API currently allows all origins (`*`). For production, restrict this to your frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Model Files

The API uses pre-trained models located in the `models/` folder:
- `models/classifier.pkl` - Trained Random Forest model
- `models/fertilizer.pkl` - Fertilizer label encoder

## Testing

Test the API with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 25,
    "Humidity": 70,
    "Moisture": 45,
    "Soil_Type": "Loamy",
    "Crop_Type": "Wheat",
    "Nitrogen": 45,
    "Potassium": 30,
    "Phosphorous": 35
  }'
```

## Frontend Integration

Update your frontend service to use the deployed API URL:

```typescript
// In src/services/mlApiService.ts
constructor() {
  // Change this to your deployed API URL
  this.baseUrl = 'https://your-api-url.railway.app';
}
```

## Monitoring

The API includes logging for:
- Model loading status
- Prediction requests
- Error handling

For production monitoring, consider:
- Application Performance Monitoring (APM) tools
- Log aggregation services
- Health check endpoints
- Metrics collection
