
# Insurance Cross Sell Prediction 🏠🏥
[![GitHub](https://img.shields.io/badge/GitHub-code-blue?style=flat&logo=github&logoColor=white&color=red)](https://github.com/prsdm/mlops-project) [![Medium](https://img.shields.io/badge/Medium-view_article-green?style=flat&logo=medium&logoColor=white&color=green)](https://medium.com/@prasadmahamulkar/machine-learning-operations-mlops-for-beginners-a5686bfe02b2)

Welcome to the Insurance Cross-Selling Prediction project! This documentation is designed to help beginners walk through the entire process, from setting up the project environment to training a machine learning model and deploying it. The goal of this project is to predict which customers are most likely to purchase additional insurance products using a machine learning model.


## Diagram
Below is the architecture diagram that illustrates the flow of the project from data ingestion to model deployment:
![Image](mlops.jpg)

## Get Started
To get started with the project, follow the steps below:

#### 1. Clone the Repository
Clone the project repository from GitHub:
```bash
git clone https://github.com/prsdm/ml-project.git
```
```bash
cd ml-project
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
docker build -t my_fastapi .
```
```bash
docker run -p 80:80 my_fastapi
```
Once your Docker image is built, you can push it to Docker Hub, making it accessible for deployment on any cloud platform.
#### 7. Monitor the Model
Integrate Evidently AI to monitor the model for data drift and performance degradation:

```bash
run monitor.ipynb file
```
