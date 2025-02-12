from fastapi import FastAPI
app=FastAPI()


def get_app_description():
    return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
	)


@app.get("/")
async def root():
    return {"message": get_app_description()}


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#Load the Iris Dataset
iris = load_iris()
X, Y= iris.data, iris.target

#Train a logisticregression model
model = LogisticRegression()
model.fit(X,Y)

#Define a function to predict the species'
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features =[[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return iris.target_names[prediction[0]]

#Define the pydantic model for inputb data
from pydantic import BaseModel

class IrisData(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

#Create Api endpoint for prediction
@app.post("/predict")
async def predict_species(iris_data:IrisData):
    species =predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)   
    return {"species": species} 