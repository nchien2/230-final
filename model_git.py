import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.git.modeling_git import (
    GitPreTrainedModel,
    GitEmbeddings, 
    GitEncoder, 
    GitProjection,
    _expand_mask,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
)


class GitModel(GitPreTrainedModel):
#class GitModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = GitEmbeddings(config)
        #self.image_encoder = GitVisionModel(config.vision_config)
        self.encoder = GitEncoder(config)

        self.visual_projection = GitProjection(config)

        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
                for _ in range(config.num_image_with_embedding)
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full(
            (num_memory, num_tgt + past_key_values_length),
            float("-inf"),
            device=tgt.device,
            dtype=dtype,
        )
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )

        if past_key_values_length > 0:
            tgt_mask = torch.zeros(
                (tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length),
                dtype=dtype,
                device=tgt_mask.device,
            )

        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError("Memory key padding mask must be a boolean tensor.")
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float("-inf")
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt)
        )
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]

        return full_attention_mask

    #@add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #pixel_values: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import requests
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")
        >>> model = AutoModel.from_pretrained("microsoft/git-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = "this is an image of two cats"

        >>> inputs = processor(text, images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        projected_visual_features = None
        # replace pixel values with visual_embeds as the following steps:
        # visual_features = [] # need to figure out the embed dims and whether we can reshape to 2d
        # for frame_idx in range(visual_embeds.shape[1]):
        #     visual_features_frame = visual_embeds[:, frame_idx, :]
        #     visual_features.append(visual_features_frame)
        # finally, concatenate all features along sequence dimension
        #visual_features = torch.cat(visual_features, dim=1)
        visual_features = visual_embeds
        projected_visual_features = self.visual_projection(visual_features)
        
        # comment below steps that convert "pixel_values" to "visual_features"
        # if pixel_values is not None:
        #     if pixel_values.ndim == 4:
        #         # here we assume pixel_values is of shape (batch_size, num_channels, height, width)
        #         visual_features = self.image_encoder(pixel_values).last_hidden_state
 
        #     elif pixel_values.ndim == 5:
        #         # here we assume pixel_values is of shape (batch_size, num_frames, num_channels, height, width)
        #         visual_features = []
        #         for frame_idx in range(pixel_values.shape[1]):
        #             visual_features_frame = self.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
        #             visual_features_frame += self.img_temperal_embedding[frame_idx]
        #             visual_features.append(visual_features_frame)

        #         # finally, concatenate all features along sequence dimension
        #         visual_features = torch.cat(visual_features, dim=1)

        #     else:
        #         raise ValueError("pixel_values must be of rank 4 or 5")

        #     projected_visual_features = self.visual_projection(visual_features)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if projected_visual_features is None:
            projected_visual_features = torch.zeros(
                (embedding_output.shape[0], 0, embedding_output.shape[2]),
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )

        # Repeat visual features to match embedding batch size.
        projected_visual_features = projected_visual_features.repeat(
            embedding_output.size(0) // projected_visual_features.size(0), 1, 1
        )

        # concatenate patch token and text token embeddings
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)

        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)

        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_visual_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(
                embedding_output.device
            )
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :] += expanded_attn_mask

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=combined_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            #pixel_values_present=pixel_values is not None,
            pixel_values_present=visual_embeds is not None,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
        
        
class GitForCausalLM(GitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.git = GitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    #@add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #pixel_values: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        Video captioning example:

        ```python
        >>> import av
        >>> import numpy as np
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download
        >>> from transformers import AutoProcessor, AutoModelForCausalLM

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

        >>> # set seed for reproducability
        >>> np.random.seed(45)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # load video
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample frames
        >>> num_frames = model.config.num_image_with_embedding
        >>> indices = sample_frame_indices(
        ...     clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
        ... )
        >>> frames = read_video_pyav(container, indices)

        >>> pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values

        >>> generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

        >>> print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))
        Generated caption: ['a woman is sitting at a table and she is talking about the food she is holding.']
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.git(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            #pixel_values=pixel_values,
            visual_embeds=visual_embeds,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            #"pixel_values": kwargs.get("pixel_values", None),
            "visual_embeds": kwargs.get("visual_embeds", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

