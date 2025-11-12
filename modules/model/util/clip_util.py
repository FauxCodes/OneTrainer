from typing import List

import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        text_encoder_output = text_encoder(
            tokens,
            attention_mask=attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )

        pooled_text_encoder_output = None
        if add_pooled_output:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_text_encoder_output = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_text_encoder_output = text_encoder_output.pooler_output

        text_encoder_output = text_encoder_output.hidden_states[default_layer - layer_skip] if add_output else None

        if add_layer_norm and text_encoder_output is not None:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(text_encoder_output)

    return text_encoder_output, pooled_text_encoder_output

def encode_clip_long(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
        expanded_chunk_size : int = 75,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        if tokens is not None:
            token_groups = tokens
        else:
            token_groups = None

        text_encoder_output = text_encoder(
            token_groups,
            attention_mask=attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_state = text_encoder_output.hidden_states[default_layer - layer_skip]
        # transform (chunk_count, chunk_size + 2, N) -> (chunk_count * chunk_size, N)
        hidden_state_content = hidden_state[:, 1:-1, :]
        hidden_state_content = hidden_state_content.reshape((-1, hidden_state_content.shape[-1]))
        # slice hidden_state for <BOS> and <EOS> tokens, reshape to (1,N)
        # assemble <BOS> <content> <EOS> and reshape to (chunk_size * chunk_count + 2,N)
        hidden_state = torch.cat([
            hidden_state[0, 0, :].unsqueeze(0),
            hidden_state_content.unsqueeze(0)[0, :, :],
            hidden_state[0, -1, :].unsqueeze(0)
        ]).squeeze()

        pooled_state = None
        if add_pooled_output:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_state = text_encoder_output.text_embeds
                pooled_state = pooled_state.mean(dim=0).reshape((1,pooled_state.shape[-1]))
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_state = text_encoder_output.pooler_output.mean(dim=0)

        text_encoder_output = text_encoder_output.hidden_states[default_layer - layer_skip] if add_output else None

        if add_layer_norm and text_encoder_output is not None:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            hidden_state = final_layer_norm(hidden_state)

        return hidden_state, pooled_state
    else:
        return text_encoder_output, pooled_text_encoder_output


def tokenize_chunked(text: str, tokenizer: CLIPTokenizer, chunk_size: int = 75) -> dict:
    chunks = _chunk_prompt(text, tokenizer, chunk_size)
    print("Chunked prompt: ", chunks)

    if not text:
        return tokenizer(
            text,
            max_length=chunk_size + 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    tokenized_chunks = []
    for chunk in chunks:
        tokenized = tokenizer(
            chunk,
            max_length=chunk_size + 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokenized_chunks.append(tokenized)

    return {
        'input_ids': torch.cat([chunk['input_ids'] for chunk in tokenized_chunks], dim=0),
        'attention_mask': torch.cat([chunk['attention_mask'] for chunk in tokenized_chunks], dim=0),
        'num_chunks': len(tokenized_chunks)
    }

def _chunk_prompt(text: str, tokenizer: CLIPTokenizer, chunk_size: int = 75) -> List[str]:
    tokens = tokenizer.encode(text)
    content_tokens = tokens[1:-1] if tokens[0] == tokenizer.bos_token_id else tokens

    chunks = []
    curr_chunk = []

    for token in content_tokens:
        if len(curr_chunk) < chunk_size:
            curr_chunk.append(token)
        else:
            chunk_text = tokenizer.decode(curr_chunk)
            if ', ' in chunk_text:
                parts = chunk_text.rsplit(', ', 1)
                if len(parts) == 2:
                    first_part, remaining = parts
                    if first_part:
                        chunks.append(first_part + ',')
                    curr_chunk = tokenizer.encode(remaining)[1:-1]
                    curr_chunk.append(token)
                    continue

            chunks.append(chunk_text)
            curr_chunk = [token]

    if curr_chunk:
        chunks.append(tokenizer.decode(curr_chunk))

    return chunks
