import sys
import os
from dotenv import load_dotenv
import torch
import tiktoken

sys.path.append(os.path.abspath(os.path.join('.')))

load_dotenv()

from src.pretraining import create_streaming_dataloader
from src.pretraining import train_llm
from src.gpt import GPTModel

MODEL_SAVE_PATH = "data/models/tinystories_model.pth"
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "qkv_bias": False,
    "drop_rate_mha": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_embedding": 0.1
}
START_CONTEXT = "Every effort moves you"
DATASET_NAME = "roneneldan/TinyStories"
EXPERIMENT_NAME = "pretrain_tinystories"

def main() -> None:
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataloader = create_streaming_dataloader(DATASET_NAME, max_length=256, stride=512, batch_size=8)
    valid_dataloader = create_streaming_dataloader(DATASET_NAME, max_length=256, stride=512, batch_size=8, train_set=False)

    track_train_loss, track_val_loss, track_tokens_seen, track_lrs = train_llm(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        num_epoch=1,
        start_context=START_CONTEXT,
        warmup_steps=20,
        tokenizer=tokenizer,
        eval_iter=50,
        eval_freq=100,
        initial_warmup=0.000001,
        peak_lr=0.0001,
        save_path=MODEL_SAVE_PATH,
        mlflow_experiment_name=EXPERIMENT_NAME,
    )

if __name__ == "__main__":
    main()