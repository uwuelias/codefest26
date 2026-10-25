# Human Sound

Human Sound is a full-stack audio classification app that helps determine whether a spoken recording is likely produced by a human or by an AI system. It combines a FastAPI backend, a React/Vite frontend, and a Wav2Vec2-based embedding model with a lightweight logistic-regression classifier.

## Overview

The app accepts an uploaded audio clip (.wav, .m4a, or .mp3), extracts speech embeddings, runs the classifier, and returns a verdict:

- Real
- Fake
- Inconclusive

It is designed to be a simple demo and proof-of-concept for audio authenticity detection.

## Features

- Audio upload with client-side validation
- FastAPI backend for inference
- Wav2Vec2 feature extraction from pretrained speech model
- Logistic regression scoring for human-vs-AI classification
- Confidence score and probability output
- Clean web UI for quick testing

## Tech Stack

- Backend: Python, FastAPI, PyTorch, Transformers, librosa, NumPy
- Frontend: React, TypeScript, Vite
- Model: Wav2Vec2 embeddings + logistic regression classifier

## Project Structure

```text
.
├── README.md
├── backend/
│   ├── build.py
│   ├── main.py
│   ├── pyproject.toml
│   ├── train.py
│   ├── uv.lock
│   ├── cache/
│   ├── routers/
│   │   └── predict.py
│   ├── services/
│   │   ├── extract.py
│   │   └── predict.py
│   └── weights/
│       ├── classifier.pt
│       └── scaler.joblib
├── frontend/
│   ├── README.md
│   ├── index.html
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── pnpm-workspace.yaml
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       └── components/
│           └── AudioSubmissionForm.tsx
```

## Prerequisites

- Python 3.13+
- Node.js 20+
- pnpm
- uv (recommended for Python environment management)

## Quick Start

### 1) Backend setup

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```text
http://localhost:8000
```

### 2) Frontend setup

In a separate terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

The frontend will run at:

```text
http://localhost:5173
```

### 3) Configure the frontend API URL

The app reads an environment variable named `VITE_API_URL`.

Example:

```bash
cd frontend
export VITE_API_URL=http://localhost:8000
pnpm dev
```

If `VITE_API_URL` is not set, it falls back to the default local backend URL used by the app.

## API

### Endpoint

```http
POST /api/predict/
```

### Request

Send an uploaded audio file using multipart form data:

- `file`: audio file

Supported formats:

- `.wav`
- `.m4a`
- `.mp3`

### Example response

```json
{
  "verdict": "fake",
  "confidence": 0.87,
  "p_fake": 0.87
}
```

### Response fields

- `verdict`: `real`, `fake`, or `inconclusive`
- `confidence`: how confident the model is in its prediction
- `p_fake`: estimated probability that the audio is AI-generated

## Model Pipeline

The backend does the following:

1. Receives the uploaded audio file.
2. Normalizes the clip to 16kHz mono audio.
3. Extracts Wav2Vec2 hidden-state embeddings.
4. Scales the embedding using the trained scaler.
5. Applies a logistic regression classifier.
6. Converts the model score to a probability and verdict.

The trained model weights live in:

- `backend/weights/classifier.pt`
- `backend/weights/scaler.joblib`

## Training Notes

The project includes a training script at `backend/train.py` that:

- loads saved embeddings from the cache
- standardizes the feature vectors
- trains a logistic regression model in PyTorch
- saves the classifier and scaler for inference

## Notes

- The backend enforces a temporary file workflow when processing uploads.
- The frontend validates the file type and size before submitting.
- The app is best treated as a local demo or research prototype, not a production-grade forensic system.

## Future Improvements

Possible extensions include:

- Recognize AI-generated music
- Increase accuracy
- User larger, more diverse datasets
- URL support
- Train on unfiltered and compressed audio
- Show visualization of what part sounds AI-generated
- More audio file formats


## AI Usage

- To generate `README.md`
- Debugging
- Advice on improving our model
