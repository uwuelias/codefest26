from pathlib import Path
import joblib
import numpy as np
import torch
from routers import predict
from torch import nn

from services.extract import get_embedding

WEIGHTS = Path(__file__).parent / "weights"
UNSURE = (0.4,0.6) #inconclusivity range

#reloads classifier
classifier = nn.Linear(768, 1)
classifier.load_state_dict(torch.load(WEIGHTS / "classifier.pt"))
classifier.eval()
scaler = joblib.load(WEIGHTS / "scaler.joblib")

#audio -> probability
def predict(path):
    embedding = get_embedding(path)
    x = scaler.transform(embedding.reshape(1,-1))
    x = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        logit = classifier(x)
    return torch.sigmoid(logit).item()

#audio -> predict -> result
def analyze(path):
    p_fake = predict(path)
    if UNSURE[0] <= p_fake <= UNSURE[1]:
        verdict = "inconclusive"
    else:
        verdict = "fake" if p_fake > 0.5 else "real"
    return {
        "verdict": verdict,
        "confidence": round(max(p_fake, 1-p_fake), 3),
        "p_fake": round(p_fake, 3),
    }