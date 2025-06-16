import cv2
import json
import boto3
import numpy as np
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from config import SURFACE_DISEASE_LABELS, RETINOBLASTOMA_LABELS, DETECTION_SURFACE_TYPE, DETECTION_RETINA_TYPE, RESULTS_FOLDER
from validation_model.model import EyeClassifierCNN 
from classification_model.model import EyeDiseaseClassifierCNN
from retinoblastoma_model.model import RetinoblastomaClassifierNonBinaryCNN

from photo_transformation.dataset_prep_lib import process_single_validation_image, process_single_classification_image, process_single_retinoblastoma_image

from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

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
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(
        ..., description="Parameter to provide a timestamp of request."
    )
    

validation_model = EyeClassifierCNN()
validation_model.load_state_dict(torch.load("info/deployed_models/validation_model_release_version.pth", map_location=torch.device('cpu'), weights_only=True))
validation_model.eval()

classification_model = models.resnet18(weights=None)
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, 8)
classification_model.load_state_dict(torch.load("info/deployed_models/classification_model_RESNET_release_version.pth", map_location=torch.device('cpu'), weights_only=True))
classification_model.eval()

retinoblastoma_model = RetinoblastomaClassifierNonBinaryCNN()
retinoblastoma_model.load_state_dict(torch.load("info/deployed_models/retinoblastoma_model_release_version.pth", map_location=torch.device('cpu'), weights_only=True))
retinoblastoma_model.eval()

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
        
        # Get image from the body
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()
        
        # Transform image into tensor
        nparr = np.frombuffer(file_content, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        processed_img_cv = process_single_validation_image(img_rgb)
        processed_tensor = torch.from_numpy(processed_img_cv).float() / 255.0
        processed_tensor = (processed_tensor - 0.5) / 0.5
        processed_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)
        
        # Run validation
        output = validation_model(processed_tensor)
        predicted = int(output.item())
        is_eye = predicted == 0 # 0 is 'eye', 1 is 'not eye'

        return {"is_eye": is_eye}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/classify-surface-eye")
async def classify_surface_eye(request: ClassificationRequestModel):
    """Classify eye surface diseases."""
    url = request.url
    user_id = request.user_id
    timestamp = request.timestamp
    print(url)
    
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()
        
        # Get image from the body
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()
        
        # Transform image into tensor
        nparr = np.frombuffer(file_content, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        processed_img_cv = process_single_classification_image(img_rgb)

        processed_img_cv = np.transpose(processed_img_cv, (2, 0, 1))
        processed_tensor = torch.from_numpy(processed_img_cv).float() / 255.0
        processed_tensor = processed_tensor.unsqueeze(0)
        
        # Run validation
        with torch.no_grad():
            output = classification_model(processed_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        result = {SURFACE_DISEASE_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        
        base_path, old_folder, file_name = url.rsplit("/", 2)
        new_file_name = f"{user_id}_{DETECTION_SURFACE_TYPE}_{timestamp}.txt"
        new_s3_path = f"{base_path}/{RESULTS_FOLDER}/{new_file_name}"
        s3_key = "/".join(new_s3_path.split("/")[3:])

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps({**result, "image_url": url}),
            ContentType="application/json",
        )
        
        return {**result, "path": new_s3_path}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/classify-retina-eye")
async def classify_retina_eye(request: ClassificationRequestModel):
    """Classify eye retina diseases."""
    url = request.url
    user_id = request.user_id
    timestamp = request.timestamp
    print(url)
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()
        
        # Get image from the body
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()
        
        # Transform image into tensor
        nparr = np.frombuffer(file_content, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        processed_img_cv = process_single_retinoblastoma_image(img_rgb)
        
        processed_img_cv = np.transpose(processed_img_cv, (2, 0, 1))
        processed_tensor = torch.from_numpy(processed_img_cv).float() / 255.0
        processed_tensor = processed_tensor.unsqueeze(0)
        
        # Run validation
        with torch.no_grad():
            output = retinoblastoma_model(processed_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        result = {RETINOBLASTOMA_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        
        base_path, old_folder, file_name = url.rsplit("/", 2)
        new_file_name = f"{user_id}_{DETECTION_RETINA_TYPE}_{timestamp}.txt"
        new_s3_path = f"{base_path}/{RESULTS_FOLDER}/{new_file_name}"
        s3_key = "/".join(new_s3_path.split("/")[3:])

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps({**result, "image_url": url}),
            ContentType="application/json",
        )
        
        return {**result, "path": new_s3_path}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))