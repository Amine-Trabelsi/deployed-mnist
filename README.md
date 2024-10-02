# Mnist model deployment using docker
Web app for a deployed mnist machine learning model
convolutional neural network
## Data
To use the data it is possible to download it from the official mnist files and use the code in [data/convert_to_csv.ipynb](data/convert_to_csv.ipynb)

## Training
Code for training the model is in [code/models/model.ipynb](code/models/model.ipynb)

## Deployment
### FastAPI
fastAPI is used to create an api endpoint the takes image data and returns the prediction from the model

### Streamlit
streamlit is used to create an interface that accepts an image, performs preprocessing and calls the api endpoint.

### Docker
a docker image is used to deploy both containers
