import torch
import tiktoken
import torch


def text_to_tokens_ids(text: str, tokenizer):
    encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_text = torch.tensor(encoded_text).unsqueeze(0)
    return encoded_text


def tokens_ids_to_text(tokens: torch.tensor, tokenizer):
    flatten_text = tokens.squeeze(0)
    return tokenizer.decode(flatten_text.tolist())


def generate_simple_text(model, idx, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        current_input = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(current_input)
        last_token = logits[:, -1, :]
        prob_token = torch.softmax(last_token, dim=-1)
        idx_next = torch.argmax(prob_token, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
