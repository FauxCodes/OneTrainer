import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection


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
        clip_chunk_size: int = 75,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        if tokens is not None:
            token_groups = _group_tokens(tokens, clip_chunk_size)
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
        ])

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

        return hidden_state.unsqueeze(0), pooled_state
    else:
        return text_encoder_output, pooled_text_encoder_output

def _group_tokens(tokens: Tensor, clip_chunk_size: int = 75):
    if tokens.dim() == 2:
        tokens = tokens.squeeze(0)
    stripped_tokens = tokens[1:-1]  # slice off <EOS> and <BOS> tokens
    chunk_count = stripped_tokens.shape[0] // clip_chunk_size
    # reshape (1,N)->(C,expanded_chunk_size), where C is the number of chunks, N is a multiple of expanded_chunk_size
    stripped_tokens = stripped_tokens.reshape(chunk_count, clip_chunk_size)
    token_groups = []
    for i in range(0, chunk_count):
        # reassemble each chunk to be <BOS> <chunk_content> <EOS>
        chunk = (
            tokens[0].unsqueeze(0),
            stripped_tokens[i, :],
            tokens[-1].unsqueeze(0)
        )
        token_groups.append(torch.cat(chunk))
    return torch.stack(token_groups)
