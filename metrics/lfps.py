#
# LFPS implementation adapted for single-image evaluation in JESFormer.
# Parts of the feature extraction and checkpoint loading logic are derived from
# the RETFound-related code shipped under `metrics/retfound/`.
#
# Keep the original copyright and license notices in third-party source files.
#
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_

import metrics.retfound.models_vit as retfound_models
from metrics.retfound.pos_embed import interpolate_pos_embed

warnings.simplefilter(action='ignore', category=FutureWarning)

_RETFOUND_CACHE = {}


def _build_retfound_model(
    resume='weights/RETFound_mae_natureCFP.pth',
    model_name='RETFound_mae',
    input_size=224,
    drop_path=0.2,
    nb_classes=2,
    device=None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache_key = (resume, model_name, input_size, drop_path, nb_classes, device)
    if cache_key in _RETFOUND_CACHE:
        return _RETFOUND_CACHE[cache_key]

    if model_name == 'RETFound_mae':
        model = retfound_models.RETFound_mae(
            img_size=input_size,
            num_classes=nb_classes,
            drop_path_rate=drop_path,
        )
    else:
        raise ValueError(f'Unsupported RETFound model: {model_name}')

    checkpoint = torch.load(resume, map_location='cpu')
    checkpoint_model = checkpoint['model']
    checkpoint_model = {k.replace('backbone.', ''): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in checkpoint_model.items()}

    state_dict = model.state_dict()
    for key in ['head.weight', 'head.bias']:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _RETFOUND_CACHE[cache_key] = model
    return model


def _to_tensor_image(img, input_order='HWC', bgr_input=False):
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.dim() == 3:
            if input_order == 'HWC':
                img = img.permute(2, 0, 1)
        elif img.dim() != 2:
            raise ValueError(f'Unsupported tensor shape: {img.shape}')
        tensor = img.float()
    else:
        array = np.asarray(img)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        if input_order == 'CHW':
            array = np.transpose(array, (1, 2, 0))
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()

    if tensor.max() > 1:
        tensor = tensor / 255.0
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if bgr_input:
        tensor = tensor[[2, 1, 0], :, :]
    return tensor


def _to_tensor_mask(mask, input_order='HWC'):
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 4:
            mask = mask.squeeze(0)
        if mask.dim() == 3 and input_order == 'HWC':
            mask = mask.permute(2, 0, 1)
        mask_tensor = mask.float()
    else:
        array = np.asarray(mask)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        if input_order == 'CHW':
            array = np.transpose(array, (1, 2, 0))
        mask_tensor = torch.from_numpy(array).permute(2, 0, 1).float()

    if mask_tensor.shape[0] > 1:
        mask_tensor = mask_tensor[:1]
    if mask_tensor.max() > 1:
        mask_tensor = mask_tensor / 255.0
    return (mask_tensor > 0.5).float()


def _prepare_inputs(img1, img2, mask, input_size=224, input_order='HWC', bgr_input=False, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    img1_tensor = _to_tensor_image(img1, input_order=input_order, bgr_input=bgr_input).unsqueeze(0).to(device)
    img2_tensor = _to_tensor_image(img2, input_order=input_order, bgr_input=bgr_input).unsqueeze(0).to(device)
    mask_tensor = _to_tensor_mask(mask, input_order=input_order).unsqueeze(0).to(device)

    img1_tensor = F.interpolate(img1_tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
    img2_tensor = F.interpolate(img2_tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, size=(input_size, input_size), mode='nearest')
    return img1_tensor, img2_tensor, mask_tensor


def _selected_levels(model):
    num_blocks = len(model.blocks)
    if num_blocks == 24:
        return num_blocks, [5, 11, 17, 23]
    if num_blocks == 12:
        return num_blocks, [3, 5, 7, 11]
    raise NotImplementedError(f'Unsupported RETFound depth: {num_blocks}')


@torch.no_grad()
def calculate_lfps(
    img1,
    img2,
    mask,
    resume='weights/RETFound_mae_natureCFP.pth',
    model_name='RETFound_mae',
    input_size=224,
    drop_path=0.2,
    nb_classes=2,
    input_order='HWC',
    bgr_input=False,
    return_unmasked=False,
    device=None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_retfound_model(
        resume=resume,
        model_name=model_name,
        input_size=input_size,
        drop_path=drop_path,
        nb_classes=nb_classes,
        device=device,
    )
    img1_tensor, img2_tensor, mask_tensor = _prepare_inputs(
        img1,
        img2,
        mask,
        input_size=input_size,
        input_order=input_order,
        bgr_input=bgr_input,
        device=device,
    )

    num_blocks, selected_levels = _selected_levels(model)
    patch_size = model.patch_embed.patch_size[0]
    density = F.avg_pool2d(mask_tensor, kernel_size=patch_size, stride=patch_size).view(1, -1).unsqueeze(-1)
    weighted_mask = torch.pow(density + 0.001, 0.5)

    img1_features = model.get_intermediate_layers(x=img1_tensor, n=num_blocks)
    img2_features = model.get_intermediate_layers(x=img2_tensor, n=num_blocks)
    img1_features = [img1_features[index] for index in selected_levels]
    img2_features = [img2_features[index] for index in selected_levels]

    img1_feature = torch.cat(img1_features, dim=0)
    img2_feature = torch.cat(img2_features, dim=0)
    img1_mask_feature = img1_feature * weighted_mask
    img2_mask_feature = img2_feature * weighted_mask

    cosine_similarity = F.cosine_similarity(img1_feature.flatten(1).unsqueeze(0), img2_feature.flatten(1).unsqueeze(0), dim=1)
    masked_cosine_similarity = F.cosine_similarity(
        img1_mask_feature.flatten(1).unsqueeze(0),
        img2_mask_feature.flatten(1).unsqueeze(0),
        dim=1,
    )

    if return_unmasked:
        return float(masked_cosine_similarity.mean().item()), float(cosine_similarity.mean().item())
    return float(masked_cosine_similarity.mean().item())
