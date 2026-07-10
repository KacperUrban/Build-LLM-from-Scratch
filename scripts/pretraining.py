from scripts.text_generation import (
    generate_simple_text,
    text_to_tokens_ids,
    tokens_ids_to_text,
)
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaderv1(
    txt,
    max_length=256,
    stride=128,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def calc_loss_batch(input, target, model, device):
    input = input.to(device)
    target = target.to(device)
    output = model(input)
    loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), target.flatten())
    return loss


def calc_loss_dataloader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(len(dataloader), num_batches)

    for i, (input, target) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input, target, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_llm(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epoch,
    device,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    track_train_loss, track_val_loss, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for i in range(num_epoch):
        model.train()
        for input, target in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input, target, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(
                    train_dataloader, val_dataloader, model, device, eval_iter
                )
                track_train_loss.append(train_loss)
                track_val_loss.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {i+1}, step: {global_step:06d}")
                print(f"Train loss: {train_loss:.3f}, Valid loss: {val_loss:.3f}")
                print(
                    f"Training perplexity: {torch.exp(torch.tensor(train_loss)):.3f}, Valid perplexity: {torch.exp(torch.tensor(val_loss)):.3f}"
                )

        generate_and_print_sample(model, tokenizer, start_context, device)
    return track_train_loss, track_val_loss, track_tokens_seen


def eval_model(train_dataloader, val_dataloader, model, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_dataloader(
            val_dataloader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context, device):
    model.eval()
    context_length = model.pos_emb.weight.shape[0]
    input_tokens = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        output = generate_simple_text(model, input_tokens, 50, context_length)
    decoded_output = tokens_ids_to_text(output, tokenizer)
    print(decoded_output.replace("\n", " "))
    model.train()
