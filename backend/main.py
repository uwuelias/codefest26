import pathlib
import tempfile
import joblib
import numpy as np
import torch.nn as nn
from extract.py import get_embedding 

classifier = nn.Sequential(
    nn.Linear(768,256), nn.ReaLU(), nn.Dropout(0.3), nn.Linear(256,1),)

classifier.load_state_dict(torch.load("weights/classifier.pt"))
classifier.eval()
scaler = joblib.load("weights/scaler.joblib")

def predict(path):
    embedding = get_embedding(path)
    x = scaler.transform(embedding.reshape(1,-1))
    x = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        logit = classifier(x)
    return torch.sigmoid(logit).item()

