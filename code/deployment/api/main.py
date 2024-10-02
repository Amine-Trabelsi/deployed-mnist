from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from typing import List
import os
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # MNIST images are 28x28
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize FastAPI app
app = FastAPI()

# HOME_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = "/app/models/mnist_model.pth"

model = SimpleNN()
model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode

# Request body structure
class ImageRequest(BaseModel):
    image: List[float]  # The flattened 28x28 image data

# Prediction route
@app.post("/predict")
async def predict(image_request: ImageRequest):
    image = np.array(image_request.image).reshape(1, 28*28)  # Reshape to the expected format
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}