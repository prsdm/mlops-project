# Insurance Cross Sell Prediction 🏠🏥

An insurance company currently provides Health Insurance to its customers and now seeks to expand its offerings by cross-selling Vehicle Insurance to its existing policyholders. The primary objective of this project is to develop a machine learning model that predicts whether customers who have purchased Health Insurance in the past year would be interested in purchasing Vehicle Insurance.

## Diagram
Below is the architecture diagram that illustrates the flow of the project from data ingestion to model deployment:
![Image](docs/mlops.jpg)

## Get Started
To get started with the project, follow the steps below:

#### 1. Clone the Repository
Clone the project repository from GitHub:

```bash
git clone https://github.com/yourusername/insurance-cross-selling-prediction.git
cd insurance-cross-selling-prediction
```
#### 2. Set Up the Environment
Ensure you have Python 3.8+ installed. Create a virtual environment and install the necessary dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Alternatively, you can use the Makefile command:
```bash
make setup
```
#### 3. Data Preparation
Pull the data from DVC. If this command doesn't work, the train and test data are already present in the data folder:
```bash
dvc pull
```

#### 4. Train the Model
To train the model, run the following command:

```bash
python main.py 
```
Or use the Makefile command:

```bash
make run
```
This script will load the data, preprocess it, train the model, and save the trained model to the models/ directory.

#### 5. FastAPI
Start the FastAPI application by running:

```bash
uvicorn app:app --reload
```

#### 6. Docker
To build the Docker image and run the container:

```bash
docker build -t my_app .
docker run -p 80:80 my_app
```
#### 7. Monitor the Model
Integrate Evidently AI to monitor the model for data drift and performance degradation:

```bash
run monitor.ipynb
```
If this doesn't work, you can use Jupyter Notebook to run the monitoring script.

