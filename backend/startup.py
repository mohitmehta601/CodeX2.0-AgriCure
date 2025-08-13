#!/usr/bin/env python3
"""
Startup script to ensure the ML model is properly trained and ready
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import fastapi
        import uvicorn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def setup_model():
    """Setup and train the ML model"""
    try:
        print("🔄 Setting up ML model...")
        
        # Run the model training script
        result = subprocess.run([sys.executable, "retrain_model.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Model setup completed successfully")
            print(result.stdout)
            return True
        else:
            print("❌ Model setup failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error setting up model: {e}")
        return False

def start_api():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting ML API server...")
        os.system("python run_local.py")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    print("🌱 AgriCure ML API Startup")
    print("=" * 40)
    
    if not check_dependencies():
        print("\n💡 Install missing dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    if not setup_model():
        print("\n💡 Model setup failed. Check the dataset and try again.")
        sys.exit(1)
    
    print("\n🎉 Setup complete! Starting API server...")
    start_api()