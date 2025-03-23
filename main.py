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
from validation_model.model import EyeClassifierCNN 

app = FastAPI()

class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])
])

validation_model = EyeClassifierCNN()
validation_model.load_state_dict(torch.load("models_info/fixed_seed_model_4.pth", weights_only=True))
validation_model.eval()

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
        
        transformed_img = transform(img).unsqueeze(0)
        
        output = validation_model(transformed_img)
        predicted = int(output.item())
        is_eye = predicted == 0  # 0 is 'eye', 1 is 'not eye'

        return {"value": is_eye}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))