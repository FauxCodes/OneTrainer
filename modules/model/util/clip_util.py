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


def encode_clip_chunked(
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

        max_length = text_encoder.config.max_position_embeddings
        chunk_size = max_length - 2

        if tokens.shape[1] > max_length:
            bos_token = tokens[:, 0:1]
            eos_token = tokens[:, -1:]
            content_tokens = tokens[:, 1:-1]

            outputs = []
            pooled_outputs = []

            for i in range(0, content_tokens.shape[1], chunk_size):
                chunk = content_tokens[:, i:i + chunk_size]

                chunk_attention_mask = None
                if use_attention_mask and attention_mask is not None:
                    # chunking the content part of the attention mask
                    # attention_mask[0] is for BOS, attention_mask[-1] is for EOS
                    # chunking content (1:-1)
                    content_mask = attention_mask[:, 1:-1]
                    chunk_mask = content_mask[:, i:i + chunk_size]

                    if chunk_mask.shape[1] < chunk_size:
                        # pad attention mask with 0 (ignored)
                        padding_mask = torch.zeros((chunk_mask.shape[0], chunk_size - chunk_mask.shape[1]), device=chunk_mask.device, dtype=chunk_mask.dtype)
                        chunk_mask = torch.cat([chunk_mask, padding_mask], dim=1)

                    # reconstruct chunk attention mask with BOS (1) and EOS (1)
                    chunk_attention_mask = torch.cat([
                        attention_mask[:, 0:1],
                        chunk_mask,
                        attention_mask[:, -1:]
                    ], dim=1)

                if chunk.shape[1] < chunk_size:
                    padding = eos_token.repeat(1, chunk_size - chunk.shape[1])
                    chunk = torch.cat([chunk, padding], dim=1)

                chunk_tokens = torch.cat([bos_token, chunk, eos_token], dim=1)

                chunk_output, chunk_pooled_output = encode_clip(
                    text_encoder=text_encoder,
                    tokens=chunk_tokens,
                    default_layer=default_layer,
                    layer_skip=layer_skip,
                    add_output=add_output,
                    add_pooled_output=add_pooled_output,
                    use_attention_mask=use_attention_mask,
                    attention_mask=chunk_attention_mask,
                    add_layer_norm=add_layer_norm,
                )

                outputs.append(chunk_output)
                pooled_outputs.append(chunk_pooled_output)

            text_encoder_output = torch.cat(outputs, dim=1) if add_output else None
            pooled_text_encoder_output = pooled_outputs[-1] if add_pooled_output else None
        else:
            text_encoder_output, pooled_text_encoder_output = encode_clip(
                text_encoder=text_encoder,
                tokens=tokens,
                default_layer=default_layer,
                layer_skip=layer_skip,
                add_output=add_output,
                text_encoder_output=text_encoder_output,
                add_pooled_output=add_pooled_output,
                pooled_text_encoder_output=pooled_text_encoder_output,
                use_attention_mask=use_attention_mask,
                attention_mask=attention_mask,
                add_layer_norm=add_layer_norm,
            )

    return text_encoder_output, pooled_text_encoder_output
