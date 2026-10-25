from pathlib import Path

import av
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

#layers in wav2vec2 that we found to be optimal for accuracy out of the 12
LAYERS = [4,5,6]

#load wav2vec2 pretrained no labels
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()


def _load_m4a(audio_path):
    chunks = []
    with av.open(audio_path) as container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            raise ValueError("No audio stream found")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio_stream):
            chunks.extend(
                output.to_ndarray().reshape(-1)
                for output in resampler.resample(frame)
            )
        chunks.extend(
            output.to_ndarray().reshape(-1)
            for output in resampler.resample(None)
        )

    if not chunks:
        raise ValueError("No audio samples found")
    return np.concatenate(chunks).astype(np.float32, copy=False)


#normalize audio to 16khz, 1 channel and output array of numbers
def get_embedding(audio_path):
    if Path(audio_path).suffix.lower() == ".m4a":
        speech = _load_m4a(audio_path)
    else:
        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad(): 
        outputs = model(**inputs, output_hidden_states=True)
    layers = torch.stack([outputs.hidden_states[i].mean(dim=1).squeeze(0) for i in LAYERS])
    return layers.mean(dim=0).numpy()
