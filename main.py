import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torch.nn as nn
import uvicorn

# Load the trained model
MODEL_PATH = "cnn_model.pth"
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 300
class_names = ['Bungalow', 'High-rise', 'Storey-Building']

# Define your model (Must match the architecture used for training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8), 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, len(class_names))
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create FastAPI instance
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}, Content type: {file.content_type}")

        # Check if file is empty
        file_bytes = await file.read()
        if not file_bytes:
            return {"error": "Uploaded file is empty"}

        # Try opening the image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        print("Image opened successfully!")

        # Preprocess image
        image = transform(image).unsqueeze(0)  # Add batch dimension
        print("Image transformed successfully!")

        # Make prediction
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        print(f"Predicted class: {class_names[predicted_class]}")
        return {"prediction": class_names[predicted_class]}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Azure's dynamic port if available
    uvicorn.run(app, host="0.0.0.0", port=port)

