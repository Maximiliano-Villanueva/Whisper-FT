######
# https://github.com/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb
######

from datasets import load_dataset, DatasetDict
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor

###
# load dataset
###


data_files = {"train": "dataset/train.csv", "test": "dataset/test.csv"}
common_voice = load_dataset("csv", data_files=data_files, delimiter = ';')

pretrained_model = "openai/whisper-base"
#Prepare Feature Extractor, Tokenizer and Data
tokenizer = WhisperTokenizer.from_pretrained(pretrained_model, language="Spanish", task="transcribe")
processor = WhisperProcessor.from_pretrained(pretrained_model, language="Spanish", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)

#Since our input audio is sampled at 48kHz, we need to downsample it to 16kHz

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


#Now that we've prepared our data, we're ready to dive into the training pipeline. The 🤗 Trainer will do much of the heavy lifting for us. All we have to do is:
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


####
# Evaluate
####
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


#load a pretrained checkpoint

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

for name, modules in model.named_modules():
    print(name, modules)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=f'./whisper-es',  # change to a repo name of your choice
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=2,
    max_steps=10,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=25,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)



from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)



trainer.train()

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "language": "es",
    "model_name": "Whisper Small es",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-base",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

trainer.save_model('whisper-base-es/model')
tokenizer.save_pretrained('whisper-base-es/tokenizer')
processor.save_pretrained('whisper-base-es/processor')

metric = evaluate.load("wer")






