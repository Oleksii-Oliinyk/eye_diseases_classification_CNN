import json
from urllib.parse import urlparse
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

import torchvision.models as models
import torch.nn as nn

from config import SURFACE_DISEASE_LABELS, RETINOBLASTOMA_LABELS
from validation_model.model import EyeClassifierCNN 
from classification_model.model import EyeDiseaseClassifierCNN
from retinoblastoma_model.model import RetinoblastomaClassifierNonBinaryCNN

app = FastAPI()

class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")
    
class ClassificationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")


validation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])
])

retinoblastoma_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

classifier_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

validation_model = EyeClassifierCNN()
validation_model.load_state_dict(torch.load("info/models/validation/validation_model_release_ver.pth", weights_only=True))
validation_model.eval()

retinoblastoma_model = RetinoblastomaClassifierNonBinaryCNN()
retinoblastoma_model.load_state_dict(torch.load("info/models/retinoblastoma/retinoblastoma_model_release_version.pth", weights_only=True))
retinoblastoma_model.eval()

classification_model = models.resnet18(pretrained=False)
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, 8)
classification_model.load_state_dict(torch.load("info/models/classification/classification_model_RESNET_release_version.pth", weights_only=True))
classification_model.eval()



@app.post("/validate-eye")
async def validate_skin(request: ValidationRequestModel):
    url = request.url
    print(url)

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content))
        
        transformed_img = validation_transform(img).unsqueeze(0)
        
        output = validation_model(transformed_img)
        predicted = int(output.item())
        is_eye = predicted == 0  # 0 is 'eye', 1 is 'not eye'

        return {"is_eye": is_eye}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.post("/classify-surface-eye")
async def classify_skin(request: ClassificationRequestModel):
    url = request.url

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = classifier_transform(img).unsqueeze(0)

        with torch.no_grad():
            output = classification_model(img_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        result = {SURFACE_DISEASE_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        return {**result}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/retinoblastoma-eye")
async def classify_skin(request: ClassificationRequestModel):
    url = request.url

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = retinoblastoma_transform(img).unsqueeze(0)

        with torch.no_grad():
            output = retinoblastoma_model(img_tensor)
            print(output)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        result = {RETINOBLASTOMA_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        return {**result}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))