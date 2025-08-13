// Service to interact with the FastAPI ML backend
export interface FertilizerPredictionInput {
  Temperature: number;
  Humidity: number;
  Moisture: number;
  Soil_Type: string;
  Crop_Type: string;
  Nitrogen: number;
  Potassium: number;
  Phosphorous: number;
}

export interface FertilizerPredictionOutput {
  fertilizer: string;
  confidence?: number;
  model_accuracy?: number;
}

export interface ModelInfo {
  model_type: string;
  model_accuracy?: number;
  features_count?: number;
  available_soil_types: string[];
  available_crop_types: string[];
  available_fertilizers: string[];
}

class MLApiService {
  private baseUrl: string;

  constructor() {
    // Use environment variable or fallback to localhost
    this.baseUrl = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';
    this.logger.info(`ML API URL: ${this.baseUrl}`);
  }

  private logger = {
    info: (msg: string) => console.log(`[MLApiService] ${msg}`),
    error: (msg: string) => console.error(`[MLApiService] ${msg}`)
  };

  async getPrediction(input: FertilizerPredictionInput): Promise<FertilizerPredictionOutput> {
    try {
      this.logger.info(`Making prediction request to ${this.baseUrl}/predict`);
      this.logger.info(`Input data: ${JSON.stringify(input)}`);
      
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });

      if (!response.ok) {
        const errorData = await response.json();
        this.logger.error(`API error: ${response.status} - ${errorData.detail}`);
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      this.logger.info(`Prediction result: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      this.logger.error(`Error getting prediction: ${error}`);
      throw error;
    }
  }

  async getModelInfo(): Promise<ModelInfo> {
    try {
      this.logger.info(`Getting model info from ${this.baseUrl}/model-info`);
      const response = await fetch(`${this.baseUrl}/model-info`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      this.logger.info(`Model info: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      this.logger.error(`Error getting model info: ${error}`);
      throw error;
    }
  }

  async healthCheck(): Promise<{status: string; model_loaded: boolean; available_fertilizers: string[]}> {
    try {
      this.logger.info(`Health check to ${this.baseUrl}/health`);
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      this.logger.info(`Health check result: ${JSON.stringify(data)}`);
      return {
        status: data.status || 'unknown',
        model_loaded: data.model_loaded || false,
        available_fertilizers: data.available_fertilizers || []
      };
    } catch (error) {
      this.logger.error(`Health check failed: ${error}`);
      return {
        status: 'unhealthy',
        model_loaded: false,
        available_fertilizers: []
      };
    }
  }

  // Helper method to validate input ranges
  validateInput(input: FertilizerPredictionInput): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check for required fields
    if (typeof input.Temperature !== 'number') errors.push('Temperature is required');
    if (typeof input.Humidity !== 'number') errors.push('Humidity is required');
    if (typeof input.Moisture !== 'number') errors.push('Moisture is required');
    if (!input.Soil_Type) errors.push('Soil Type is required');
    if (!input.Crop_Type) errors.push('Crop Type is required');
    if (typeof input.Nitrogen !== 'number') errors.push('Nitrogen is required');
    if (typeof input.Potassium !== 'number') errors.push('Potassium is required');
    if (typeof input.Phosphorous !== 'number') errors.push('Phosphorous is required');

    if (input.Temperature < 0 || input.Temperature > 50) {
      errors.push('Temperature must be between 0 and 50°C');
    }

    if (input.Humidity < 0 || input.Humidity > 100) {
      errors.push('Humidity must be between 0 and 100%');
    }

    if (input.Moisture < 0 || input.Moisture > 100) {
      errors.push('Moisture must be between 0 and 100%');
    }

    if (input.Nitrogen < 0 || input.Nitrogen > 200) {
      errors.push('Nitrogen must be between 0 and 200');
    }

    if (input.Potassium < 0 || input.Potassium > 200) {
      errors.push('Potassium must be between 0 and 200');
    }

    if (input.Phosphorous < 0 || input.Phosphorous > 200) {
      errors.push('Phosphorous must be between 0 and 200');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}

export const mlApiService = new MLApiService();
export default mlApiService;