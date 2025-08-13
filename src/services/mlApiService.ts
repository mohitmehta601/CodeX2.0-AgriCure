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
}

export interface ModelInfo {
  model_type: string;
  available_soil_types: string[];
  available_crop_types: string[];
  available_fertilizers: string[];
}

class MLApiService {
  private baseUrl: string;

  constructor() {
    // Use localhost for development, change this to your deployed API URL
    this.baseUrl = 'http://localhost:8000';
  }

  async getPrediction(input: FertilizerPredictionInput): Promise<FertilizerPredictionOutput> {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting prediction:', error);
      throw error;
    }
  }

  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/model-info`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting model info:', error);
      throw error;
    }
  }

  async healthCheck(): Promise<{status: string; model_loaded: boolean; available_fertilizers: string[]}> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
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