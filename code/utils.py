import yaml
import numpy as np
import SimpleITK as sitk
import torch
from easydict import EasyDict
from config_schema import validate_config



def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return EasyDict(cfg)


def convert_cuda(item, device=None):
    device = device or 'cuda'
    return {key: value.to(device=device, dtype=torch.float32)
            if torch.is_tensor(value) and not (key.endswith('view_ids') or key.endswith('mask')) else
            value.to(device) if torch.is_tensor(value) else value
            for key, value in item.items()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sitk_load(path, uint8=False, spacing_unit='mm', return_direction=False):
    # load as float32
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    if return_direction:
        return image, spacing, origin, np.array(itk_img.GetDirection())
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False, direction=None):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64)) # unit: mm
    if direction is not None:
        out.SetDirection(np.asarray(direction, dtype=np.float64).tolist())
    sitk.WriteImage(out, path)
