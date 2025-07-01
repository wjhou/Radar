import ast
import math
from einops import rearrange, repeat
from einops_exts import rearrange_many
from einops import rearrange
from PIL import Image
import torch
from torch import einsum, nn


from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from transformers import CLIPVisionModel
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel
from transformers import PretrainedConfig, logging, CONFIG_MAPPING
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

logger = logging.get_logger(__name__)


def fixed_cross_entropy(
    source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(
        shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


class XGenMMVisionEncoderConfig(PretrainedConfig):
    model_type = "xgenmm_vision_encoder"

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        anyres_grids: list[int] = [
            [384, 768],
            [768, 384],
            [768, 768],
            [1152, 384],
            [384, 1152],
        ],
        **kwargs,
    ):
        self.model_name = model_name
        self.anyres_grids = anyres_grids
        super().__init__(**kwargs)


class XGenMMVisionTokenizerConfig(PretrainedConfig):
    model_type = "xgenmm_vision_tokenizer"

    def __init__(
        self,
        vis_feature_dim: int = 1152,
        lang_embedding_dim: int = 3072,
        num_vis_tokens: int = 128,
        image_aspect_ratio: str = "anyres",
        **kwargs,
    ):
        self.vis_feature_dim = vis_feature_dim
        self.lang_embedding_dim = lang_embedding_dim
        self.num_vis_tokens = num_vis_tokens
        self.image_aspect_ratio = image_aspect_ratio
        super().__init__(**kwargs)


class XGenMMConfig(PretrainedConfig):
    model_type = "xgenmm"

    def __init__(
        self,
        vision_encoder_config: dict = None,
        vision_tokenizer_config: dict = None,
        text_config: dict = None,
        **kwargs,
    ):

        if vision_encoder_config is None:
            vision_encoder_config = {
                "image_aspect_ratio": "anyres",
                "anyres_patch_sampling": True,
            }
            logger.info(
                "vision_encoder_config is None. initializing the XGenMMVisionEncoderConfig with default values."
            )

        if vision_tokenizer_config is None:
            vision_tokenizer_config = {}
            logger.info(
                "vision_tokenizer_config is None. Initializing the XGenMMVisionTokenizerConfig with default values."
            )

        if text_config is None:
            text_config = {
                "initial_tokenizer_len": 32012,
                "pad_token_id": 32011,
                "bos_token_id": 1,
                "eos_token_id": 32000,
                "vocab_size": 32064,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
                "attention_dropout": 0.0,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "original_max_position_embeddings": 4096,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-05,
                "use_cache": True,
                "rope_theta": 10000.0,
                "rope_scaling": None,
                "sliding_window": 2047,
                "return_dict": True,
                "output_hidden_states": False,
                "output_attentions": False,
                "torchscript": False,
                "torch_dtype": "bfloat16",
                "use_bfloat16": False,
                "tf_legacy_loss": False,
                "pruned_heads": {},
                "tie_word_embeddings": False,
                "chunk_size_feed_forward": 0,
                "is_encoder_decoder": False,
                "is_decoder": False,
                "cross_attention_hidden_size": None,
                "add_cross_attention": False,
                "tie_encoder_decoder": False,
                "max_length": 20,
                "min_length": 0,
                "do_sample": False,
                "early_stopping": False,
                "num_beams": 1,
                "num_beam_groups": 1,
                "diversity_penalty": 0.0,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "typical_p": 1.0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "encoder_no_repeat_ngram_size": 0,
                "bad_words_ids": None,
                "num_return_sequences": 1,
                "output_scores": False,
                "return_dict_in_generate": False,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
                "remove_invalid_values": False,
                "exponential_decay_length_penalty": None,
                "suppress_tokens": None,
                "begin_suppress_tokens": None,
                "finetuning_task": None,
                "id2label": {0: "LABEL_0", 1: "LABEL_1"},
                "label2id": {"LABEL_0": 0, "LABEL_1": 1},
                "tokenizer_class": None,
                "prefix": None,
                "bos_token_id": 1,
                "pad_token_id": 32000,
                "eos_token_id": 32000,
                "sep_token_id": None,
                "decoder_start_token_id": None,
                "task_specific_params": None,
                "problem_type": None,
                "model_type": "phi3",
                "_attn_implementation": "flash_attention_2",
            }
            logger.info(
                "text_config is None. Initializing the text config with default values (`Phi3Config`)."
            )

        self.vision_encoder_config = XGenMMVisionEncoderConfig(**vision_encoder_config)

        self.vision_tokenizer_config = XGenMMVisionTokenizerConfig(
            **vision_tokenizer_config
        )

        text_model_type = (
            text_config["model_type"] if "model_type" in text_config else "phi3"
        )
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        for key in ["initial_tokenizer_len", "pad_token_id"]:
            if key not in self.text_config.to_dict():
                raise ValueError(f"The key `{key}` is missing in the text_config.")

        super().__init__(**kwargs)


def hasattr_recursive(obj, att):
    """
    Check if obj has nested attribute
    Example: hasattr_recursive(obj, 'a.b.c') is equivalent to hasattr(obj, 'a') and hasattr(obj.a, 'b') and hasattr(obj.a.b, 'c')
    """
    if att == "":
        return True
    i = att.find(".")
    if i < 0:
        return hasattr(obj, att)
    else:
        try:
            return hasattr_recursive(getattr(obj, att[:i]), att[i + 1 :])
        except:
            return False


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def check_embedding_fns(lang_model):
    """Checks for and attempts to set {get/set}_{input/output}_embeddings functions to the model"""
    if not has_fn(lang_model, "get_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.get_input_embeddings = lambda: lang_model.transformer.wte
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.get_input_embeddings = lambda: lang_model.decoder.embed_tokens
        else:
            raise ValueError(
                "We require the language encoder to have a get_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "transformer.wte", x
            )
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "model.decoder.embed_tokens", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "get_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.get_output_embeddings = lambda: lang_model.lm_head
        else:
            raise ValueError(
                "We require the language encoder to have a get_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.set_output_embeddings = lambda x: setattr_recursive(
                lang_model, "lm_head", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )


def has_fn(model, fn_name):
    """Check if model has a function fn_name"""
    return callable(getattr(model, fn_name, None))


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


def unpad_image(tensor, original_size, keep_original_shape=False):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        if keep_original_shape:
            attention_mask = torch.ones(
                (current_height, current_width), device=tensor.device
            )
            attention_mask[:padding, :] = 0
            attention_mask[current_height - padding :, :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
            return unpadded_tensor, None
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        if keep_original_shape:
            attention_mask = torch.ones(
                (current_height, current_width), device=tensor.device
            )
            attention_mask[:, :padding] = 0
            attention_mask[:, current_width - padding :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, :, padding : current_width - padding]
            return unpadded_tensor, None


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # FIXME: determine grid_pinpoints from image sizes.
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    processor_size = processor.transforms[0].size
    patches = divide_to_patches(image_padded, processor_size[0])

    image_original_resize = image.resize((processor_size[0], processor_size[0]))

    image_patches = [image_original_resize] + patches
    image_patches = [processor(image_patch) for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat(
            (x, latents), dim=-2
        )  # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat(
                (
                    vision_attn_masks,
                    torch.ones(
                        (latents.shape[0], latents.shape[-2]),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                ),
                dim=-1,
            )
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros(
                (q.size(0), 1, 1, q.size(-2), k.size(-2)),
                dtype=q.dtype,
                device=q.device,
            )
            vision_attn_masks = repeat(
                vision_attn_masks, "b n -> b 1 1 l n", l=q.size(-2)
            )
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def num_params(module, filter_to_trainable=False):
    """Returns the number of parameters in the module, or optionally only the trainable parameters"""
    if filter_to_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=128,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, vision_attn_masks):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents

        if exists(self.projection):
            return self.projection(self.norm(latents))
        else:
            return self.norm(latents)


class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        max_original_id: int,
        num_additional_embeddings: int = 0,
        _weight: torch.Tensor = None,
        num_original_embeddings: int = None,
        embedding_dim: int = None,
        partially_freeze=True,
        device=None,
        dtype=None,
        pad_token_id=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`):
                The largest token id that should be embedded using the regular embedding (regular `weight`).
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal self.weight.shape[0]
            num_additional_embeddings (`int`):
                Number of additional tokens to initialize an Embedding matrix for (`additional_weight`).
            _weight (`torch.Tensor`, *optional*, defaults to `None`): The regular weight tensor.
                If provided, this sets the `num_original_embeddings` and `embedding_dim` parameters.
            num_original_embeddings (`int`):
                self.weight.shape[0]
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `True`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # validate args
        if pad_token_id is not None and pad_token_id > max_original_id:
            raise ValueError(
                f"pad_token_id must be <= max_original_id. Got {pad_token_id} and {max_original_id}."
                + "If the original tokenizer does not have a pad_token_id, use pad_token_id=None."
            )
        if _weight is not None:
            assert (num_original_embeddings is None) or (
                _weight.shape[0] == num_original_embeddings
            ), f"num_original_embeddings={num_original_embeddings} but _weight.shape[0]={_weight.shape[0]}"
            assert (embedding_dim is None) or (
                _weight.shape[1] == embedding_dim
            ), f"embedding_dim={embedding_dim} but _weight.shape[1]={_weight.shape[1]}"
            num_original_embeddings = _weight.shape[0]
            embedding_dim = _weight.shape[1]
        else:
            assert (
                num_original_embeddings is not None
            ), "num_original_embeddings must be provided if _weight is not provided"
            assert (
                embedding_dim is not None
            ), "embedding_dim must be provided if _weight is not provided"

        super().__init__(
            num_embeddings=num_original_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=pad_token_id,
            _weight=_weight,
        )
        self.max_original_id = max_original_id
        self.padding_idx = pad_token_id
        self.num_additional_embeddings = num_additional_embeddings
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        self.additional_embedding.requires_grad_(require_additional_grad)

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
        embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids > self.max_original_id)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(
            input_ids_additional_vocab - self.max_original_id - 1
        )

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_original_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.max_original_id + 1,
            self.num_additional_embeddings,
            self.embedding_dim,
            (not self.weight.requires_grad),
        )


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `additional_out_features` > 0,
    then it will create `additional_out_features * in_features` additional parameters that are always trained. If
    `additional_out_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        max_original_id: int,
        additional_out_features: int = 0,
        _weight: torch.Tensor = None,
        _bias: torch.Tensor = None,
        in_features: int = None,
        original_out_features: int = None,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`): The largest token id that should be extracted from the regular weight.
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal original_out_features - 1
            _weight: torch.Tensor, *optional*, defaults to `None`. The regular weight tensor.
                If provided, this sets the `in_features` and `original_out_features` parameters.
            _bias: torch.Tensor, *optional*, defaults to `None`. The regular bias tensor.
            in_features: int. Input hidden size.
            original_out_features: int. Original out_features of the language model's get_output_embeddings() function.
            additional_out_features: int. Number of additional trainable dimensions.
            bias: bool. Whether to include a bias term.
            partially_freeze: bool, *optional*, defaults to `True`): If `True`, the regular `weight` will be frozen.
        """
        # argument validation
        if _weight is not None:
            assert (_weight.shape[0] == original_out_features) or (
                original_out_features is None
            ), f"original_out_features={original_out_features} but _weight.shape[0]={_weight.shape[0]}"
            assert (_weight.shape[1] == in_features) or (
                in_features is None
            ), f"in_features={in_features} but _weight.shape[1]={_weight.shape[1]}"
            in_features = _weight.shape[1]
            original_out_features = _weight.shape[0]
        else:
            assert (
                in_features is not None
            ), "in_features must be provided if _weight is not provided"
            assert (
                original_out_features is not None
            ), "original_out_features must be provided if _weight is not provided"

        if _bias is not None:
            assert bias is True, "bias must be True if _bias is provided"

        # initialize original linear
        super().__init__(in_features, original_out_features, bias, device, dtype)

        # set weight and bias manually
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        if _bias is not None:
            self.bias = nn.Parameter(_bias)

        self.in_features = in_features
        self.original_out_features = original_out_features
        self.max_original_id = max_original_id

        # initialize additional linear
        self.additional_out_features = additional_out_features
        self.has_bias = bias
        if additional_out_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=additional_out_features,
                bias=self.has_bias,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        if self.has_bias:
            self.bias.requires_grad_(require_regular_grad)
        self.additional_fc.requires_grad_(require_additional_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        output = output[..., : self.max_original_id + 1]

        if self.additional_out_features > 0:
            additional_features = F.linear(
                input, self.additional_fc.weight, self.additional_fc.bias
            )
            output = torch.cat((output, additional_features), -1)
        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, additional_out_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.max_original_id + 1,
            self.additional_out_features,
            self.bias is not None,
            (not self.weight.requires_grad or not self.bias.requires_grad),
        )


class VLM(nn.Module):
    """
    Generic vision-language model (VLM) class.
    A VLM consists of four components:
        1. A vision encoder that extracts features from pixels, e.g. CLIP
            input: (B, T_img, F, C, H, W)
            output: (B, T_img, F, v, d)
        2. A vision tokenizer that converts these features to visual token-like embeddings, e.g. Perceiver, or a linear projection head
            input: (B, T_img, F, v, d)
            output: (B, T_img, n, d)
        3. A fusion method that allows the language model to attend to these tokens, e.g. cross-attention, or placing the tokens directly in the language model's input sequence
        4. A language model
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): e.g. CLIP
            vision_tokenizer (nn.Module): e.g. PerceiverResampler
            lang_model (nn.Module): e.g. MPT
            initial_tokenizer_len (int): size of the original tokenizer vocab
            pad_token_id (int): id of the pad token
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # save dimension information
        self.lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        if hasattr(lang_model.config, "d_model"):
            self.lang_hidden_dim = lang_model.config.d_model  # mpt uses d_model
        else:
            self.lang_hidden_dim = lang_model.config.hidden_size
        self.vis_embedding_dim = vision_tokenizer.dim_media
        self.num_tokens_per_vis = vision_tokenizer.num_tokens_per_media

        # core components
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        self.lang_model = lang_model

        # lm embeddings
        self.pad_token_id = pad_token_id
        self.initial_tokenizer_len = initial_tokenizer_len
        input_embeds = DecoupledEmbedding(
            max_original_id=initial_tokenizer_len - 1,
            num_additional_embeddings=len(self.special_tokens),
            _weight=self.lang_model.get_input_embeddings().weight,
            pad_token_id=self.pad_token_id,
        ).to(self.lang_model.dtype)
        if hasattr(input_embeds, "additional_embedding"):
            input_embeds.additional_embedding.weight.data.normal_(
                mean=0.0,
                std=(
                    self.lang_model.config.initializer_range
                    if hasattr(self.lang_model.config, "initializer_range")
                    else 0.02
                ),
            )
        self.lang_model.set_input_embeddings(input_embeds)

        out_embeds = DecoupledLinear(
            max_original_id=initial_tokenizer_len - 1,
            additional_out_features=len(self.special_tokens),
            _weight=self.lang_model.get_output_embeddings().weight,
            _bias=(
                self.lang_model.get_output_embeddings().bias
                if hasattr(self.lang_model.get_output_embeddings(), "bias")
                else None
            ),
        ).to(self.lang_model.dtype)
        if hasattr(out_embeds, "additional_fc"):
            out_embeds.additional_fc.weight.data.normal_(
                mean=0.0,
                std=(
                    self.lang_model.config.initializer_range
                    if hasattr(self.lang_model.config, "initializer_range")
                    else 0.02
                ),
            )
        self.lang_model.set_output_embeddings(out_embeds)

        # gradient checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features, None)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def _encode_vision_x_anyres(self, samples, device):
        assert self.anyres_grids is not None
        image_raw = samples[
            "image"
        ]  # list of patch list in of shape [1, N_patch, C, H, W]
        image_sizes = samples["image_size"]

        # Image_raw can be a list of list of patches, when a `samples` has multiple images.
        if isinstance(image_raw[0], list):
            images = [x.squeeze(0) for sample_img in image_raw for x in sample_img]
            image_sizes = [s for sample_sizes in image_sizes for s in sample_sizes]
        else:
            # assert isinstance(image_raw[0], torch.Tensor), f"Unkown image type: {image_raw[0]}"
            # concate list of patches into one big patch for any res encoding.
            images = [x.squeeze(0) for x in image_raw]  # [N_patch, C, H, W]
        image = torch.cat(images, dim=0)  # [\sum{B}{N_patch_i}, C, H, W]
        image = image.to(device)

        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                image_embeds = self.vision_encoder.trunk.forward_features(image)
            elif self.vision_encoder.__class__.__name__ in [
                "CLIPVisionModel",
                "SiglipVisionTransformer",
            ]:
                image_embeds = self.vision_encoder(image).last_hidden_state
            else:
                image_embeds = self.vision_encoder(image)[1]  # OpenCLIP returns tuples

        if isinstance(self.vision_encoder, CLIPVisionModel) or isinstance(
            self.vision_encoder, SiglipVisionTransformer
        ):
            base_img_size = self.vision_encoder.config.image_size
        else:
            base_img_size = self.vision_encoder.image_size[0]

        if self.vision_encoder.__class__.__name__ == "TimmModel":
            grid_size = self.vision_encoder.trunk.patch_embed.grid_size
        elif self.vision_encoder.__class__.__name__ in [
            "CLIPVisionModel",
            "SiglipVisionTransformer",
        ]:
            grid_size_base = (
                self.vision_encoder.config.image_size
                // self.vision_encoder.config.patch_size
            )
            grid_size = (grid_size_base, grid_size_base)
        else:
            grid_size = self.vision_encoder.grid_size
        height, width = grid_size

        if not image_embeds.shape[1] == height * width:
            assert (
                image_embeds.shape[1] == height * width + 1
            )  # For vision encoders that has [CLS] token.
            image_embeds = image_embeds[:, 1:, :]  # Drop the cls token for each patch.
        n_vis_token_per_patch = image_embeds.shape[1]

        # Split encoded patches and merge patch features
        # 1. Get the raw sizes from samples, and split the image embeds [\sum_{B}(N_patch_i), N_tok(16*16), C]
        split_sizes = [image.shape[0] for image in images]
        image_embeds = torch.split(image_embeds, split_sizes, dim=0)
        # 2. For each image (consist of a list of patches), merge the patches spatially (of shape [C, n_patch_height, n_patch_width])
        new_image_embeds = []
        patch_attn_masks = []
        max_n_img_token = -1
        for idx, patch_embeds in enumerate(image_embeds):
            if patch_embeds.shape[0] > 1:
                # 3. Flatten the patch features and get [C, n_patch_height * (n_patch_width+1)]
                base_patch_embeds = patch_embeds[
                    0
                ]  # TODO: prepend the CLS token for th base patch embeds (of the resized entire image).
                patch_embeds = patch_embeds[1:]

                assert height * width == base_patch_embeds.shape[0]

                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[idx], self.anyres_grids, base_img_size
                )  # Hardcoded grid_pinpoints.
                patch_embeds = patch_embeds.view(
                    num_patch_height, num_patch_width, height, width, -1
                )

                patch_embeds = patch_embeds.permute(4, 0, 2, 1, 3).contiguous()
                patch_embeds = patch_embeds.flatten(1, 2).flatten(2, 3)
                patch_embeds, patch_attn_mask = unpad_image(
                    patch_embeds, image_sizes[idx], self.anyres_patch_sampling
                )
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat(
                        (
                            patch_embeds,
                            self.image_newline[:, None, None].expand(
                                *patch_embeds.shape[:-1], 1
                            ),
                        ),
                        dim=-1,
                    )
                if self.anyres_patch_sampling:
                    patch_embeds = patch_embeds.view(
                        -1, num_patch_height, num_patch_width, height * width
                    )
                    patch_embeds = patch_embeds.flatten(1, 2).permute(1, 2, 0)
                    assert patch_attn_mask is not None
                    patch_attn_mask = patch_attn_mask.view(
                        num_patch_height, num_patch_width, height * width
                    )
                    patch_attn_mask = patch_attn_mask.flatten(0, 1)
                    patch_embeds = torch.cat(
                        (base_patch_embeds.unsqueeze(0), patch_embeds), dim=0
                    )
                    patch_attn_mask = torch.cat(
                        (
                            torch.ones(
                                n_vis_token_per_patch, device=patch_embeds.device
                            ).unsqueeze(0),
                            patch_attn_mask,
                        ),
                        dim=0,
                    )
                else:
                    patch_embeds = patch_embeds.flatten(1, 2).transpose(0, 1)
                    patch_embeds = torch.cat((base_patch_embeds, patch_embeds), dim=0)
            else:
                patch_embeds = (
                    patch_embeds[0].unsqueeze(0)
                    if self.anyres_patch_sampling
                    else patch_embeds[0]
                )
                patch_attn_mask = (
                    torch.ones(
                        n_vis_token_per_patch, device=patch_embeds.device
                    ).unsqueeze(0)
                    if self.anyres_patch_sampling
                    else None
                )
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat(
                        (patch_embeds, self.image_newline[None]), dim=0
                    )
            if not self.anyres_patch_sampling:
                max_n_img_token = max(patch_embeds.shape[0], max_n_img_token)

            new_image_embeds.append(patch_embeds)
            patch_attn_masks.append(patch_attn_mask)

        if self.anyres_patch_sampling:
            # Return individual patches for independent token downsampling.
            return new_image_embeds, patch_attn_masks

        # 4. Pad and concat the list of image_embeds [N_tok_i, C] together into a batch. Also modify the query attention mask.
        image_embeds = []
        image_atts = []
        for image_embed in new_image_embeds:
            n_img_token = image_embed.shape[0]
            img_attn = torch.ones(
                (max_n_img_token), dtype=torch.long, device=image_embed.device
            )
            if n_img_token < max_n_img_token:
                padded_embed = torch.zeros(
                    (max_n_img_token, image_embed.shape[-1]),
                    dtype=image_embed.dtype,
                    device=image_embed.device,
                )
                padded_embed[:n_img_token, :] = image_embed
                img_attn[n_img_token:] = 0  # Mask out the padded entries.
            else:
                padded_embed = image_embed
            image_embeds.append(padded_embed)
            image_atts.append(img_attn)
        image_embeds = torch.stack(
            image_embeds, dim=0
        )  # Shape [B, N_tok_longest, C_dim]
        image_atts = torch.stack(image_atts, dim=0)  # Shape [B, N_tok_longest, C_dim]
        # TODO: reshape image_embeds and image_atts to "b T F v d"
        image_embeds = image_embeds[:, None, None, :, :]
        # image_atts = image_atts[:, None, None, :, :]

        return image_embeds, image_atts

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        # with torch.no_grad():
        # if self.vision_encoder.__class__.__name__ == "TimmModel":
        #     vision_x = self.vision_encoder.trunk.forward_features(vision_x)
        # elif self.vision_encoder.__class__.__name__ in [
        #     "CLIPVisionModel",
        #     "SiglipVisionTransformer",
        # ]:
        # if True:
        # with torch.no_grad():
        #     vision_x = self.vision_encoder(vision_x).last_hidden_state
        # else:
        # vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = self.vision_encoder(vision_x).last_hidden_state
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _concat_vision_cache(
        self, lang_x, vision_tokens, past_vision_tokens, past_media_locations, use_cache
    ):
        """
        Helper function to include the past vision tokens and past media locations in the output.
        """
        if use_cache:
            if past_media_locations is not None and past_vision_tokens is not None:
                if vision_tokens is not None:
                    updated_vision_tokens = torch.cat(
                        [
                            past_vision_tokens,
                            vision_tokens,
                        ],
                        dim=1,
                    )
                else:
                    updated_vision_tokens = past_vision_tokens
                updated_media_locations = torch.cat(
                    [
                        past_media_locations,
                        lang_x == self.media_token_id,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = vision_tokens
                updated_media_locations = lang_x == self.media_token_id

        else:
            updated_vision_tokens = None
            updated_media_locations = None

        return updated_vision_tokens, updated_media_locations

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features, None)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        output = self.lang_model.generate(
            **new_inputs,
            past_key_values=past_key_values,
            num_beams=num_beams,
            use_cache=True,
            **kwargs,
        )
        self._post_forward_hook()
        return output

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    def set_trainable(self):
        """
        Freeze appropriate parameters in the model.
        """
        raise NotImplementedError

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if self._should_apply_weight_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
        return params_with_wd, params_without_wd

    def _should_apply_weight_decay(self, parameter_name):
        """
        Return whether weight decay should be applied to a parameter.
        """
        raise NotImplementedError

    @property
    def special_tokens(self):
        """
        Returns a dict mapping from the attribute name of a special token to its string format,
         e.g. "media_token": "<image>"
        """
        assert (
            "media_token" in self._special_tokens
        ), "VLMs need to request that the tokenizer add a media_token and call set_special_token_ids to set self.media_token_id"
        return self._special_tokens

    @property
    def special_token_ids(self):
        """
        Returns a list of the special token ids
        """
        return [getattr(self, f"{att_name}_id") for att_name in self.special_tokens]

    def set_special_token_ids(self, string_to_ids):
        """
        Args:
            string_to_ids (dict): mapping from token string to id
        """
        assert set(self.special_tokens.values()).issubset(set(string_to_ids.keys()))
        for att_name, token_str in self.special_tokens.items():
            token_id = string_to_ids[token_str]
            setattr(self, f"{att_name}_id", token_id)
            setattr(self.lang_model, f"{att_name}_id", token_id)

    def init_gradient_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointWrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        from functools import partial

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )


@dataclass
class VLMOutputWithPast(CausalLMOutputWithPast):
    """
    VLMOutputWithPast is a wrapper around CausalLMOutputWithPast that adds the following attributes:
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
    """

    past_media_locations: Optional[torch.Tensor] = None
    past_vision_tokens: Optional[torch.Tensor] = None


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class VLMWithLanguageStream(VLM):
    """
    VLM that fuses modalities by inserting vision tokens directly into the language stream.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder_layers_attr_name = decoder_layers_attr_name
        if decoder_layers_attr_name is not None:
            for block in getattr_recursive(
                self.lang_model, self.decoder_layers_attr_name
            ):
                block._use_gradient_checkpointing = gradient_checkpointing

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "left",
        num_beams: int = 1,
    ):
        """
        Insert the vision tokens directly into the language stream/
        This requires us to modify the input_ids, attention_mask, and labels.
        """
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            assert attention_mask.shape[1] == past_len + lang_x.shape[1], (
                "Attention_mask must be as long as the entire past len (including image tokens) and current input IDs. "
                + "Check that you've expanded the attention mask to account for past image tokens."
            )

        if vision_tokens is None:
            return {
                "input_ids": lang_x,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # get the language embeddings
        lang_embeds = self.lang_model.get_input_embeddings()(lang_x)

        # build up the multimodal embeddings
        B = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None
        for i in range(B):
            # get index of <image> tokens in lang_x[i]
            image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]

            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                multimodal_attention_mask.append(attention_mask[i].clone())
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            # loop through the image_token_idxs and insert the vision tokens
            new_embed = lang_embeds[i].clone()
            new_attention_mask = (
                attention_mask[i].clone() if attention_mask is not None else None
            )
            if has_labels:
                new_label = labels[i].clone()

            for img_num in range(len(image_token_idxs)):
                img_idx = image_token_idxs[img_num]
                # Get vision token attention mask for padded llava-style any resolution image tokens.
                if self.image_aspect_ratio == "anyres":
                    num_vis_tokens = vision_tokens[i][img_num].shape[0]
                    if vision_attention_mask is not None:
                        vis_attention_mask = vision_attention_mask[i]
                    else:
                        vis_attention_mask = torch.ones(
                            num_vis_tokens, dtype=torch.long
                        ).to(attention_mask.device)
                else:
                    assert (
                        vision_tokens[i][img_num].shape[0] == self.num_tokens_per_vis
                    ), f"vision token number mismatch: image embedding ({vision_tokens[i][img_num].shape[0]}) \
                            vs. model.num_tokens_per_vis ({self.num_tokens_per_vis})"
                    # By default, vision tokens are not padded.
                    num_vis_tokens = self.num_tokens_per_vis
                    vis_attention_mask = torch.ones(
                        num_vis_tokens, dtype=torch.long
                    ).to(attention_mask.device)

                # Offset the rest of image tokens with current num_vis_tokens
                for j in range(img_num + 1, len(image_token_idxs)):
                    image_token_idxs[j] += num_vis_tokens - 1

                new_embed = torch.cat(
                    (
                        new_embed[:img_idx],
                        vision_tokens[i][img_num],
                        new_embed[img_idx + 1 :],
                    ),
                    dim=0,
                )
                new_attention_mask = torch.cat(
                    (
                        new_attention_mask[:img_idx],
                        vis_attention_mask,
                        new_attention_mask[img_idx + 1 :],
                    ),
                    dim=0,
                )
                if has_labels:
                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(num_vis_tokens, dtype=torch.long).to(
                                labels.device
                            )
                            * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )
            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)
        # stack
        multimodal_embeds = stack_with_padding(
            multimodal_embeds,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        multimodal_attention_mask = stack_with_padding(
            multimodal_attention_mask,
            padding_value=0,
            padding_side=padding_side,
        )
        if has_labels:
            multimodal_labels = stack_with_padding(
                multimodal_labels,
                padding_value=-100,
                padding_side=padding_side,
            )
        multimodal_inputs = {
            "inputs_embeds": multimodal_embeds,
            "attention_mask": multimodal_attention_mask,
            "labels": multimodal_labels,
        }
        return multimodal_inputs

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        # Include the past vision tokens and past media locations in the output
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )

        # return logits that are the same shape as the original input_ids
        logits = output.logits
        batch_logits = []
        B, T_txt = lang_x.shape
        for i in range(B):
            sequence_logits = []
            logits_j = 0
            for j in range(T_txt):
                if lang_x[i, j] != self.media_token_id:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += self.num_tokens_per_vis
            sequence_logits = torch.stack(sequence_logits, dim=0)  # (B, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(batch_logits, dim=0)  # (B, T_txt, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == (B, T_txt)

        # assemble the output
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=batch_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        pass

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Language model: {num_params(self.lang_model):,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True):,} trainable parameters",
            ]
        )


class XGenMMPerceiver(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        image_aspect_ratio: str = "anyres",
        anyres_patch_sampling: bool = True,
        anyres_grids: list[int] = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        # lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling
        self.anyres_grids = anyres_grids

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        prior_vision_x: Optional[torch.Tensor] = None,
        has_prior: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features, None)

            if (
                prior_vision_x is not None
                and has_prior is not None
                and has_prior.sum() > 0
            ):
                temp = prior_vision_x[has_prior == 1]
                prior_vision_features = self._encode_vision_x(vision_x=temp)
                prior_vision_tokens = self.vision_tokenizer(prior_vision_features, None)
                # vision_tokens = torch.cat((prior_vision_tokens, vision_tokens), dim=-2)
                new_vision_tokens = []
                count = 0
                for index, hp in enumerate(has_prior):
                    vtokens = vision_tokens[index]
                    if hp.item() == 1:
                        prior_vtokens = prior_vision_tokens[count]
                        count += 1
                        vtokens = torch.cat([prior_vtokens, vtokens], dim=-2)
                    new_vision_tokens.append(vtokens)
                vision_tokens = new_vision_tokens

            if hasattr(self, "observation_tokenizer") and observation_mask is not None:
                new_vision_tokens = []
                vision_attention_mask = []
                observation_tokens = self.observation_tokenizer(vision_features, None)
                for index in range(len(vision_tokens)):
                    vtokens = vision_tokens[index]
                    base_tokens = vtokens.mean(dim=1, keepdim=True)
                    otokens = observation_tokens[index] - base_tokens
                    vattention_mask = torch.ones_like(vtokens[0, :, 0])
                    oattention_mask = observation_mask[index]
                    vtokens = torch.cat([otokens, vtokens], dim=-2)
                    vattention_mask = torch.cat(
                        [oattention_mask, vattention_mask], dim=-1
                    )
                    new_vision_tokens.append(vtokens)
                    vision_attention_mask.append(vattention_mask)
                vision_tokens = new_vision_tokens
        else:
            vision_tokens = None
        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )

        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        prior_vision_x: Optional[torch.Tensor] = None,
        has_prior: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        attention_mask: torch.Tensor = None,
        observation_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features, None)

            if (
                prior_vision_x is not None
                and has_prior is not None
                and has_prior.sum() > 0
            ):
                temp = prior_vision_x[has_prior == 1]
                prior_vision_features = self._encode_vision_x(vision_x=temp)
                prior_vision_tokens = self.vision_tokenizer(prior_vision_features, None)

                new_vision_tokens = []
                count = 0
                for index, hp in enumerate(has_prior):
                    vtokens = vision_tokens[index]
                    if hp.item() == 1:
                        prior_vtokens = prior_vision_tokens[count]
                        count += 1
                        vtokens = torch.cat([prior_vtokens, vtokens], dim=-2)
                    new_vision_tokens.append(vtokens)
                vision_tokens = new_vision_tokens

            if hasattr(self, "observation_tokenizer") and observation_mask is not None:
                new_vision_tokens = []
                vision_attention_mask = []
                observation_tokens = self.observation_tokenizer(vision_features, None)
                for index in range(len(vision_tokens)):
                    vtokens = vision_tokens[index]
                    base_tokens = vtokens.mean(dim=1, keepdim=True)
                    otokens = observation_tokens[index] - base_tokens
                    vattention_mask = torch.ones_like(vtokens[0, :, 0])
                    oattention_mask = observation_mask[index]
                    vtokens = torch.cat([otokens, vtokens], dim=-2)
                    vattention_mask = torch.cat(
                        [oattention_mask, vattention_mask], dim=-1
                    )
                    new_vision_tokens.append(vtokens)
                    vision_attention_mask.append(vattention_mask)
                vision_tokens = new_vision_tokens
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output


class XGenMMVisionEncoder(PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = XGenMMVisionEncoderConfig

    def __init__(self, config: XGenMMVisionEncoderConfig):
        super().__init__(config)
        if config.model_name != "google/siglip-so400m-patch14-384":
            raise ValueError(
                f"Unsupported model {config.model_name}. New vision models will be added soon."
            )
        self.model = AutoModel.from_pretrained(config.model_name)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # assert pixel_values.ndim == 4, f"Expected 4D tensor (bs, c, h, w), got {pixel_values.ndim}"
        return self.model.encode_image(pixel_values)


# vision tokenizer
class XGenMMVisionTokenizer(PreTrainedModel):
    config_class = XGenMMVisionTokenizerConfig

    def __init__(self, config: XGenMMVisionTokenizerConfig):
        super().__init__(config)
        self.model = PerceiverResampler(
            dim=config.vis_feature_dim,
            dim_inner=config.lang_embedding_dim,
            num_latents=config.num_vis_tokens,
        )

    def forward(self, vision_features: torch.Tensor, vision_attn_masks: torch.Tensor):
        return self.model(vision_features, vision_attn_masks)


# XGenMM model
class XGenMMModelForConditionalGeneration(PreTrainedModel):
    config_class = XGenMMConfig
    base_model_prefix = "blip3"
    # supports_gradient_checkpointing = True
    _no_split_modules = ["ResidualAttentionBlock", "Phi3DecoderLayer"]
    _supports_flash_attn_2 = True
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True

    def __init__(self, config: XGenMMConfig):
        super().__init__(config)

        # vision encoder initialization
        vision_encoder = AutoModel.from_pretrained(
            config.vision_encoder_config.model_name,
            torch_dtype=config.text_config.torch_dtype,
        ).vision_model

        # language model initialization
        language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            torch_dtype=config.text_config.torch_dtype,
            attn_implementation="flash_attention_2",
        )
        check_embedding_fns(language_model)
        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]

        # vision tokenizer initialization
        if (
            config.vision_tokenizer_config.lang_embedding_dim
            != language_model.get_input_embeddings().weight.shape[1]
        ):
            overwrite = language_model.get_input_embeddings().weight.shape[1]
            config.vision_tokenizer_config.lang_embedding_dim = overwrite
            print(
                f"Warning: The language embedding dimension in the vision tokenizer config is different from the language model's embedding dimension. Overwriting the language embedding dimension in the vision tokenizer config to {overwrite}."
            )

        vision_tokenizer = XGenMMVisionTokenizer(
            config.vision_tokenizer_config
        ).model.to(language_model.dtype)

        self.vlm = XGenMMPerceiver(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=language_model,
            initial_tokenizer_len=config.text_config.initial_tokenizer_len,
            pad_token_id=config.text_config.pad_token_id,
            image_aspect_ratio=config.vision_encoder_config.image_aspect_ratio,
            anyres_patch_sampling=config.vision_encoder_config.anyres_patch_sampling,
            anyres_grids=config.vision_encoder_config.anyres_grids,
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        prior_pixel_values: Optional[torch.FloatTensor] = None,
        has_prior: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_size: Optional[Tuple] = None,
        labels: Optional[torch.LongTensor] = None,
        observation_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        outputs = self.vlm(
            lang_x=input_ids,
            attention_mask=attention_mask,
            vision_x=pixel_values,
            prior_vision_x=prior_pixel_values,
            has_prior=has_prior,
            labels=labels,
            observation_mask=observation_mask,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        prior_pixel_values=None,
        has_prior=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        self.vlm = self.vlm.eval()
        return self.vlm.generate(
            vision_x=pixel_values,
            prior_vision_x=prior_pixel_values,
            has_prior=has_prior,
            lang_x=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def update_special_tokens(self, tokenizer):
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.vlm.special_tokens.values())}
        )
        self.vlm.lang_model.config.vocab_size = len(tokenizer)
        self.vlm.set_special_token_ids(
            {
                v: tokenizer.convert_tokens_to_ids(v)
                for v in self.vlm.special_tokens.values()
            }
        )
        return tokenizer
