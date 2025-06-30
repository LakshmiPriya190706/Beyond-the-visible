import os
import pandas as pd
import numpy as np
import torch
import pydicom
from PIL import Image
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.cnn_model import TumorCNN
import glob

# Set paths
csv_folder = r"D:\csv"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
dcm_base_path = r"D:\VS dataset\manifest-1692206474218\Vestibular-Schwannoma-MC-RC"

# Load model
device = torch.device("cpu")
model = TumorCNN()
model.load_state_dict(torch.load("model/vs_tumor_confidence_model.pth", map_location=device))
model.eval()

# Preprocessing function
def predict_from_dcm(dcm_path):
    try:
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
            return model(tensor).item()
    except Exception as e:
        print(f"❌ Error reading {dcm_path}: {e}")
        return None

# Load ground truth CSV
df = pd.read_csv(csv_files[0])
predicted = []
actual = []

for _, row in df.iterrows():
    patient_id = row['patient_id']
    scan_date = row['scan_date']
    actual_score = float(row['confidence_score'])

    patient_folder = os.path.join(dcm_base_path, patient_id)
    if not os.path.exists(patient_folder):
        print(f"❌ Missing patient folder: {patient_folder}")
        continue

    found = False
    for outer in os.listdir(patient_folder):
        outer_path = os.path.join(patient_folder, outer)
        if not os.path.isdir(outer_path):
            continue

        for inner in os.listdir(outer_path):
            inner_path = os.path.join(outer_path, inner)
            if os.path.isdir(inner_path):
                dcm_files = [f for f in os.listdir(inner_path) if f.endswith(".dcm")]
                if not dcm_files:
                    continue

                mid_slice = os.path.join(inner_path, sorted(dcm_files)[len(dcm_files) // 2])
                pred = predict_from_dcm(mid_slice)
                if pred is not None:
                    predicted.append(pred)
                    actual.append(actual_score)
                    print(f"✅ Predicted {patient_id} | GT: {actual_score:.2f} | Pred: {pred:.2f}")
                    found = True
                    break
        if found:
            break

    if not found:
        print(f"⚠️ No valid DICOM found for {patient_id} on {scan_date}")

# Convert to NumPy arrays
predicted = np.array(predicted)
actual = np.array(actual)

if len(actual) == 0:
    print("❌ No predictions made. Check DICOM path and folder logic.")
    exit()

# Evaluation metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Accuracy in percentage (1 - normalized MAE)
normalized_mae = mae / (np.max(actual) - np.min(actual))
accuracy_percent = (1 - normalized_mae) * 100

print("\n📊 Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy (approx): {accuracy_percent:.2f}%")
