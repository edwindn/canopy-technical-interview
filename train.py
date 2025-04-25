import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForPreTraining
from datasets import load_dataset
from huggingface_hub import snapshot_download

wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
wav2vec2 = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")

dataset = snapshot_download("openslr/librispeech_asr")
dataset = load_dataset("openslr/librispeech_asr", split="train.100")
print(dataset)
print(dataset[0])
quit()

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