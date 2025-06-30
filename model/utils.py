import os
import torch
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms
from model.cnn_model import TumorCNN

device = torch.device("cpu")
model = TumorCNN()
model.load_state_dict(torch.load("model/vs_tumor_confidence_model.pth", map_location=device))
model.eval()

def predict_confidence(dicom_folder):
    dcm_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
    if not dcm_files:
        return 0.0

    mid = len(dcm_files) // 2
    dcm_path = os.path.join(dicom_folder, dcm_files[mid])
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = Image.fromarray((img * 255).astype(np.uint8)).convert("L")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        score = model(tensor).item()
        return round(min(max(score, 0), 100), 2)

def calculate_questionnaire_score(responses):
    total = sum(int(v) for v in responses.values())
    return round(total * 2.5, 2)

def get_stage(final_score):
    if final_score < 20:
        return "Stage 0 - No Tumor"
    elif final_score < 40:
        return "Stage I - Early"
    elif final_score < 60:
        return "Stage II - Moderate"
    elif final_score < 80:
        return "Stage III - Advanced"
    else:
        return "Stage IV - Severe"

def predict_confidence_dcm(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = Image.fromarray((img * 255).astype(np.uint8)).convert("L")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        score = model(tensor).item()
        return round(min(max(score, 0), 100), 2)

