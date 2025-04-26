import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
from dotenv import load_dotenv
import itertools
import soundfile
import librosa

load_dotenv()

hf_login(os.getenv("HF_TOKEN"))

# dataset = snapshot_download(
#     repo_id="openslr/librispeech_asr",
#     repo_type="dataset",
#     revision="main",
#     max_workers=os.cpu_count(),
# )
# dataset = load_dataset("openslr/librispeech_asr", split="test")
# print(dataset)
# print(dataset[0])

# 1. load as a stream
stream = load_dataset(
    "openslr/librispeech_asr",
    split="train.clean.100",
    streaming=True,            # ← streaming mode
)

# 2. take however many examples you actually need, e.g. 100
dataset = itertools.islice(stream, 1000)
# Convert iterator to list and create Dataset
dataset = list(dataset)
dataset = Dataset.from_list(dataset)

print(f"Created dataset with {len(dataset)} examples")
print(dataset[0]["audio"].keys())
quit()


wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
wav2vec2 = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

def map_fn(batch):
    text = batch["text"].lower()
    audio = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]

    if sr != WAV2VEC_SAMPLE_RATE:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr, 
            target_sr=WAV2VEC_SAMPLE_RATE
        )

    text_tokens = tokenizer(text).input_ids
    audio_tokens = wav2vec_processor(audio, sampling_rate=WAV2VEC_SAMPLE_RATE).input_ids

    return {"input_ids": audio_tokens, "attention_mask": [1] * len(audio_tokens), "labels": text_tokens}




print('quitting...')
quit()


llama_vocab_size = 128256
llama_sos_token = 0 ## TODO
llama_eos_token = 0
llama_pad_token = 0

WAV2VEC_SAMPLE_RATE = 16000
WAV2VEC_LATENT_DIM = 768
LLAMA_INPUT_DIM = None

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
    


model = GatedMLP(input_dim=WAV2VEC_LATENT_DIM, hidden_dim=1024, output_dim=LLAMA_INPUT_DIM)

training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("training...")
trainer.train()