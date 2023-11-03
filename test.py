from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from datasets import Audio, Dataset

import torch

# load model and processor

path_pretrained = os.path.join(os.path.dirname(__file__), 'whisper-base-es')
processor = WhisperProcessor.from_pretrained(os.path.join(path_pretrained, "processor"))
model = WhisperForConditionalGeneration.from_pretrained(os.path.join(path_pretrained, "model"))

#load test audio file
audio_path = [os.path.join(os.path.dirname(__file__), 'files', 'audio_consultas_habituales_01.wav')]
ds = Dataset.from_dict({'audio' : audio_path})
ds = ds.cast_column("audio", Audio(sampling_rate=16000))


#transcribe
input_speech = next(iter(ds))["audio"]["array"]

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "es", task = "transcribe")

input_features = processor(input_speech, return_tensors="pt").input_features 

predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)

print(transcription)
