import torch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

#load wav2vec2 pretrained no labels
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#normalize audio to 16khz, 1 channel and output array of numbers
def get_embedding(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad(): 
        hidden = model(**inputs).last_hidden_state
    return hidden.mean(dim=1).squeeze(0).numpy()

