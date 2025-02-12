FastAPI is a popular web framework for building APIs with Python. It's super simple to learn and is loved by developers.
FastAPI leverages Python type hints and is based on Pydantic. This makes it simple to define data models and request/response schemas. The framework automatically validates request data against these schemas, reducing potential errors. It also natively supports asynchronous endpoints, making it easier to build performant APIs that can handle I/O-bound operations efficiently.
Step 1: Set Up the Environment
FastAPI requires Python 3.7 or later. So make sure you have a recent version of Python installed. In the project directory, create and activate a dedicated virtual environment for the project:
$ python3 -m venv v1
$ source v1/bin/activate
Next, install the required packages. You can install FastAPI and uvicorn using pip:
$ pip3 install fastapi uvicorn
This installs FastAPI and all the required dependencies as well uvicorn, the server that we’ll use to run and test the API that we build. Because we’ll build a simple machine learning model using scikit-learn, install it in your project environment as well:
$ pip3 install scikit-learn
Step 2: Create a FastAPI App
Create a main.py file in the project directory. The first step is to create a FastAPI app instance like so:
# Create a FastAPI app
# Root endpoint returns the app description
from fastapi import FastAPI
app = FastAPI()

The Iris dataset is one of the toy datasets that you work with when starting out with data science. It has 150 data records, 4 features, and a target label (species of Iris flowers). To keep things simple, let’s create an API to predict the Iris species.
let's also define a root endpoint which returns the description of the app that we're building. To do so, we define the get_app_description function and create the root endpoint with the @app decorator like so:

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}
 
 Step 3: Build a Logistic Regression Classifier
So far we’ve instantiated a FastAPI app and have defined a root endpoint. It’s now time to do the following:
-Build a machine learning model. We’ll use a logistic regression classifier. If you’d like to learn more about logistics regression
-Define a prediction function that receives the input features and uses the machine learning model to make a prediction for the species (one of setosa, versicolor, and virginica).
We build a simple logistic regression classifier from scikit-learn and define the predict_species function as shown:

# Build a logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
	features = [[sepal_length, sepal_width, petal_length, petal_width]]
	prediction = model.predict(features)
	return iris.target_names[prediction[0]]

 Step 4: Define Pydantic Model for Input Data
 Next, we should model the data that we send in the POST request. Here the input features are the length and width of the sepals and petals—all floating point values. To model this, we create an IrisData class that inherits from the Pydantic BaseModel class like so:

# Define the Pydantic model for your input data
from pydantic import BaseModel

class IrisData(BaseModel):
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float'

 Step 5: Create an API Endpoint
Now that we’ve built the classifier and have defined the predict_species function ready, we can create the API endpoint for prediction. Like earlier, we can use the @app decorator to define the /predict/ endpoint that accepts a POST request and returns the predicted species:

# Create API endpoint
@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
	return {"species": species}

 Step 6: Run the App
 You can run the app with the following command:

$ uvicorn main:app --reload

 Step 7: Test the API
Become UNSTOPPABLE with data
Become UNSTOPPABLE with data

FastAPI Tutorial: Build APIs with Python in Minutes
Want to build APIs with Python? Learn how to do so using FastAPI with this step-by-step tutorial.
By Bala Priya C, KDnuggets Contributing Editor & Technical Content Specialist on June 13, 2024 in Python
FacebookTwitterLinkedInRedditEmailShare

bala-fastapi
Image by Author

 
FastAPI is a popular web framework for building APIs with Python. It's super simple to learn and is loved by developers.

FastAPI leverages Python type hints and is based on Pydantic. This makes it simple to define data models and request/response schemas. The framework automatically validates request data against these schemas, reducing potential errors. It also natively supports asynchronous endpoints, making it easier to build performant APIs that can handle I/O-bound operations efficiently.

This tutorial will teach you how to build your first API with FastAPI. From setting up your development environment to building an API for a simple machine learning app, this tutorial takes you through all the steps: defining data models, API endpoints, handling requests, and more. By the end of this tutorial, you’ll have a good understanding of how to use FastAPI to build APIs quickly and efficiently. So let’s get started.

 


Step 1: Set Up the Environment
 

FastAPI requires Python 3.7 or later. So make sure you have a recent version of Python installed. In the project directory, create and activate a dedicated virtual environment for the project:

$ python3 -m venv v1
$ source v1/bin/activate
 

The above command to activate the virtual environment works if you’re on Linux or MacOS. If you’re a Windows user, check the docs to create and activate virtual environments.

Next, install the required packages. You can install FastAPI and uvicorn using pip:

$ pip3 install fastapi uvicorn
 

This installs FastAPI and all the required dependencies as well uvicorn, the server that we’ll use to run and test the API that we build. Because we’ll build a simple machine learning model using scikit-learn, install it in your project environment as well:

$ pip3 install scikit-learn
 

With the installations out of the way, we can get to coding! You can find the code on GitHub.

 


Step 2: Create a FastAPI App
 

Create a main.py file in the project directory. The first step is to create a FastAPI app instance like so:

# Create a FastAPI app
# Root endpoint returns the app description

from fastapi import FastAPI

app = FastAPI()
 

The Iris dataset is one of the toy datasets that you work with when starting out with data science. It has 150 data records, 4 features, and a target label (species of Iris flowers). To keep things simple, let’s create an API to predict the Iris species.

In the coming steps, we’ll build a logistic regression model and create an API endpoint for prediction. After you’ve built the model and defined the /predict/ API endpoint, you should be able to make a POST request to the API with the input features and receive the predicted species as a response.

 
fastapi-1
Iris Prediction API | Image by Author

 
Just so it’s helpful, let's also define a root endpoint which returns the description of the app that we're building. To do so, we define the get_app_description function and create the root endpoint with the @app decorator like so:

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}
 

Sending a GET request to the root endpoint returns the description.

 


Step 3: Build a Logistic Regression Classifier
 

So far we’ve instantiated a FastAPI app and have defined a root endpoint. It’s now time to do the following:

Build a machine learning model. We’ll use a logistic regression classifier. If you’d like to learn more about logistics regression, read Building Predictive Models: Logistic Regression in Python.
Define a prediction function that receives the input features and uses the machine learning model to make a prediction for the species (one of setosa, versicolor, and virginica).
 
fastapi-2
Logistic Regression Classifier | Image by Author

 
We build a simple logistic regression classifier from scikit-learn and define the predict_species function as shown:

# Build a logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
	features = [[sepal_length, sepal_width, petal_length, petal_width]]
	prediction = model.predict(features)
	return iris.target_names[prediction[0]]
 

 


Step 4: Define Pydantic Model for Input Data
 

Next, we should model the data that we send in the POST request. Here the input features are the length and width of the sepals and petals—all floating point values. To model this, we create an IrisData class that inherits from the Pydantic BaseModel class like so:

# Define the Pydantic model for your input data
from pydantic import BaseModel

class IrisData(BaseModel):
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float
 

If you need a quick tutorial on using Pydantic for data modeling and validation, read Pydantic Tutorial: Data Validation in Python Made Super Simple.

 


Step 5: Create an API Endpoint
 

Now that we’ve built the classifier and have defined the predict_species function ready, we can create the API endpoint for prediction. Like earlier, we can use the @app decorator to define the /predict/ endpoint that accepts a POST request and returns the predicted species:

# Create API endpoint
@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
	return {"species": species}
 

And it’s time to run the app!

 


Step 6: Run the App
 

You can run the app with the following command:

$ uvicorn main:app --reload
 

Here main is the name of the module and app is the FastAPI instance. The --reload flag ensures that the app reloads if there are any changes in the source code.

Upon running the command, you should see similar INFO messages:

INFO: 	Will watch for changes in these directories: ['/home/balapriya/fastapi-tutorial']
INFO: 	Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO: 	Started reloader process [11243] using WatchFiles
INFO: 	Started server process [11245]
INFO: 	Waiting for application startup.
INFO: 	Application startup complete.
…
…
 

If you navigate to "http://127.0.0.1:8000"(localhost), you should see the app description:

 
fastapi-3
App Running on localhost

 

Step 7: Test the API
 

You can now send POST requests to the /predict/ endpoint with the sepal and petal measurements—with valid values—and get the predicted species. You can use a command-line utility like cURL. Here’s an example:

curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
 

For this example request this is the expected output:

{"species":"setosa"}
