import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

#layers in wav2vec2 that we found to be optimal for accuracy out of the 12
LAYERS = [4,5,6]

#load wav2vec2 pretrained no labels
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()

#normalize audio to 16khz, 1 channel and output array of numbers
def get_embedding(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad(): 
        outputs = model(**inputs, output_hidden_states=True)
    layers = torch.stack([outputs.hidden_states[i].mean(dim=1).squeeze(0) for i in LAYERS])
    return layers.mean(dim=0).numpy()
