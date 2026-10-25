# Human Sound: Architecture

Human Sound checks whether an audio clip is AI-generated. Users upload an audio clip, and the backend uses Meta’s pretrained Wav2Vec 2.0 model to convert the audio into numerical representations. The trained model analyzes the uploaded clip, and the app displays two results: whether the audio is AI-generated and how confident the model is in its prediction.

## Stack

- Frontend: Vite, React, TypeScript
- Backend: FastAPI
- Audio processing: 
- Model: the one we trained (weights loaded once when the server starts)

## How it works

```
browser --> audio --> FastAPI --> spectrogram --> model --> { is_ai, confidence }
```

1. User picks an audio file in the browser.
2. Frontend sends it to the backend.
3. Backend decodes it, resamples it, and turns it into a spectrogram image.
4. The model scores the image.
5. Backend returns the verdict and confidence, and the UI shows them.

## Endpoints
**POST /predict**

Takes an audio file (same format), runs the whole pipeline, and returns:

```json
{ "verdict": "real", "confidence": 0.93, "p_score": 0.3 }
```

`confidence` is how sure the model is about its own answer, so `verdict: real` with `0.93` means 93% sure it's human. It is not the probability of AI.
Errors come back as `{ "detail": "message" }` with a 400 for bad or unreadable audio, 413 for files that are too big, and 415 for unsupported formats.

## Backend

The spectrogram settings (sample rate, window size, hop length, mel bins, image size, normalization) have to match training exactly. If they drift, the model will give bad answers without throwing any errors. Keep them in one config, and ideally share that code with the training scripts.

A few other things:

- Load the model once at startup, not per request.
- Don't save uploads. Process them and delete any temp files right away.
- Set a max file size and a max duration. Longer clips can get chunked later.
- Spectrogram generation and inference are slow, so use plain `def` route handlers and let FastAPI run them in a threadpool.

## Frontend

The user flow is simple: upload, loading state, then the result card with a verdict and a confidence percentage. If something fails, show the error and let them try again.

In dev, proxy the API through Vite so we don't fight CORS. The API URL lives in `VITE_API_URL`.

## Testing

- Backend: try a known human clip and a known AI clip against both endpoints, plus some bad inputs (empty file, wrong format, too long).
- One check that the backend's spectrogram for a clip matches what the training pipeline produces for the same clip.
- Frontend: basic tests for the uploader and result card with the API mocked.

## Before going public

- Add rate limiting.
- Lock CORS down to our real domain.
- Say clearly in the UI that audio isn't stored.

## Still to decide

- Which formats to support (wav, mp3, m4a?)
- Max clip length, and whether to chunk longer audio
- Whether `/convert` returns a raw PNG or base64 JSON
- How to show results near 50/50 (a "not sure" state?)
- Upload only, or in-browser recording too
