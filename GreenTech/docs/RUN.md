# Running the GreenTech Innovations Energy Efficiency Project

This document outlines how to set up and run the project on your local machine.

## Environment Setup


```bash
# Set Python path (macOS with Homebrew Python 3.10)
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"

# Verify Python version
python3.10 --version

# Create virtual environment
python3.10 -m venv env_gt

# Activate virtual environment
source env_gt/bin/activate

# Update pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt




## Project Directory Structure



GreenTech/
├── data/
│   └── energy_efficiency.csv       # Dataset
├── src/
│   ├── main.py                     # Main execution script
│   ├── data_preprocessing.py       # Data preprocessing module
│   ├── model_training.py           # Model training module
│   └── model_deployment.py         # Deployment simulation
├── requirements.txt                # Package dependencies
└── RUN.md                          # This file
```

## Running the Project

Navigate to the project directory and activate your environment:

```bash
cd /path/to/GreenTech
source env_gt/bin/activate
```

Then run the main script:

```bash
# Run the complete workflow (training, evaluation, and demo prediction)
python src/main.py

# To run only the deployment simulation
python src/model_deployment.py
```

## Expected Output

The script will:
1. Load and preprocess the energy efficiency dataset
2. Train three models (Decision Tree, Random Forest, Neural Network) for both heating and cooling load prediction
3. Evaluate and compare model performance
4. Generate visualizations in the `output/plots` directory
5. Save trained models in the `output/models` directory
6. Demonstrate predictions on sample data

## API Service (Optional)

To start the Flask API service for real-time predictions:

1. Uncomment the last two lines in `model_deployment.py`:
```python
deployment = ModelDeployment()
deployment.run_prediction_service(port=5000)
```

2. Run the deployment script:
```bash
python src/model_deployment.py
```

3. The API will be available at `http://localhost:5000` with the following endpoints:
   - `/health` (GET): Check service health
   - `/predict` (POST): Make predictions for a single building
   - `/batch_predict` (POST): Make predictions for multiple buildings from a CSV file

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed:
```bash
pip install -r requirements.txt
```

2. Check that the dataset is in the correct location:
```bash
# The dataset should be in:
data/energy_efficiency.csv
```

3. Verify that your Python version is 3.8 or higher:
```bash
python --version
```

4. If you get "module not found" errors, make sure you're running the script from the project root directory.
