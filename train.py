import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForPreTraining, AutoModelForCausalLM, PreTrainedModel, Trainer, TrainingArguments, AutoTokenizer, Wav2Vec2Model, default_data_collator
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
from dotenv import load_dotenv
import itertools
import soundfile
import librosa
import wandb
import accelerate


WAV2VEC_SAMPLE_RATE = 16000
WAV2VEC_LATENT_DIM = 768
LLAMA_INPUT_DIM = 3200

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

wandb.init(project="audio2llama-test")

stream = load_dataset(
    "openslr/librispeech_asr",
    split="train.clean.100",
    streaming=True,
)

dataset = itertools.islice(stream, 10)
dataset = list(dataset)
dataset = Dataset.from_list(dataset)

print(f"Created dataset with {len(dataset)} examples")

wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

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
    #audio_tokens = wav2vec_processor(audio, sampling_rate=WAV2VEC_SAMPLE_RATE).input_ids

    return {"audio": audio, "audio_attention_mask": [1] * len(audio), "labels": text_tokens}

dataset = dataset.map(map_fn, batched=False, num_proc=1, remove_columns=["text", "audio"])
#dataset = dataset.with_format(type="torch", columns=["audio", "audio_attention_mask", "labels"])

# ----------------------- #

llama_vocab_size = 128256
llama_sos_token = 128000
llama_eos_token = 128001
llama_pad_token = 128001

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = F.silu(x1) * x2
        return self.proj(x)
    


projection_layer = GatedMLP(input_dim=WAV2VEC_LATENT_DIM, hidden_dim=1024, output_dim=LLAMA_INPUT_DIM)

class Audio2Llama(PreTrainedModel):
    def __init__(self):
        super().__init__(config=llama.config)

        self.processor  = wav2vec_processor
        self.wav2vec    = wav2vec2
        self.projection = projection_layer
        self.llama      = llama

    def forward(self, 
                audio: torch.Tensor,
                audio_attention_mask: torch.Tensor,
                labels: torch.Tensor):
        """
        audio: float32 waveform (batch, samples)
        audio_attention_mask: not used by wav2vec but for llama we pass later
        labels: token-ids for text (batch, tgt_len)
        """
        wav2vec_outputs = self.wav2vec(
            input_values=audio,
            return_dict=True,
        )

        feats = wav2vec_outputs.last_hidden_state

        projected = self.projection(feats)

        lm = self.llama(
            inputs_embeds=projected,
            attention_mask=audio_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return lm.loss, lm.logits
    



model = Audio2Llama()

for param in model.wav2vec.parameters():
    param.requires_grad = False

for param in model.llama.parameters():
    param.requires_grad = False

model.wav2vec.eval()
model.projection.train()
model.llama.eval()


training_args = TrainingArguments(
  output_dir="results",
  per_device_train_batch_size=1,
  gradient_accumulation_steps=1,
  learning_rate=1e-3,
  num_train_epochs=1,
  eval_steps=500,
  save_total_limit=2,
  fp16=True,
  logging_dir="logs",
  logging_steps=5,
  report_to="wandb",
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=dataset,
  data_collator=default_data_collator,
)

print("training...")
trainer.train()

print("pushing to hub...")
model.push_to_hub("audio2llama-test")