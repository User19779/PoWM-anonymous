from math import sqrt
import os.path as osp
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
import torch
import torch.nn.functional as F

from main.utils import Image, erode
from main.wmattacker import Image

import json
import cv2
import shutil


def GetDivideMethod(
    mask_tensor: torch.Tensor
) -> tuple[int, ...]:
    """_summary_

    Args:
        mask_tensor (torch.Tensor): masktensor

    Returns:
        tuple[int, ...]: (min_x, min_y, max_x, max_y),
    """
    # print(mask_tensor.shape)

    check_cover_rate = False
    if check_cover_rate:
        white_pixel_total = mask_tensor.sum().item()
        target_white_pixels = int(white_pixel_total * 0.97)

    white_positions = torch.nonzero(mask_tensor)
    min_y, min_x = white_positions[:, 1].min().item(), \
        white_positions[:, 2].min().item()
    max_y, max_x = white_positions[:, 1].max().item(), \
        white_positions[:, 2].max().item()
    ans = (min_x, min_y, max_x, max_y)
    ans = tuple(int(i) for i in ans)
    return ans


def save_adjust_image(
        img: Image.Image, mask: Image.Image, area_pos: tuple[int, ...],
        save_path: str, img_name: str, background_color=(0, 0, 0),
        A: int = 5, pad_to_certain_size: int | None = None,
        erode_kernel_size: int = 4, erode_iterations: int = 1):
    min_x, min_y, max_x, max_y = area_pos
    #  area pos 
    region_width = (max_x - min_x + 1 + 7) // 8 * 8
    region_height = (max_y - min_y + 1 + 7) // 8 * 8

    for i in range(A):
        pos = i/(A-1)
        if region_width > region_height:
            # Width is greater than height, horizontal alignment
            # Edit 0525:  pad_to_certain_size , *
            # “”
            if pad_to_certain_size is None:
                square_size = region_width
            elif isinstance(pad_to_certain_size, int):
                if pad_to_certain_size < region_width:
                    raise RuntimeError("Desired size not enough")
                square_size = pad_to_certain_size
            else:
                raise TypeError
            square_size_2 = region_width

            paste_pos_y = int(pos * (region_width - region_height))
            img_temp = Image.new(
                'RGB', (square_size, square_size), background_color)
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            img_temp.save(
                osp.join(save_path, f'{img_name}_pos_{i+1}.png'), compress_level=0)

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            # 
            # pos : (min_x,min_y,max_x,max_y), 
            pos = (0, paste_pos_y, max_x-min_x, max_y-min_y+paste_pos_y)
        else:
            # region_width <= region_height, vertical alignment
            # Edit 0525:  pad_to_certain_size , *
            # “”
            if pad_to_certain_size is None:
                square_size = region_height
            elif isinstance(pad_to_certain_size, int):
                if pad_to_certain_size < region_height:
                    raise RuntimeError("Desired size not enough")
                square_size = pad_to_certain_size
            else:
                raise TypeError
            square_size_2 = region_height

            paste_pos_x = int(pos * (region_height - region_width))
            img_temp = Image.new(
                'RGB', (square_size, square_size), background_color)
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
            img_temp.save(
                osp.join(save_path, f'{img_name}_pos_{i+1}.png'), compress_level=0)

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
            # 
            # pos : (min_x,min_y,max_x,max_y), 
            pos = (paste_pos_x, 0, max_x-min_x+paste_pos_x, max_y-min_y)

        mask_temp.save(
            osp.join(save_path, f'{img_name}_mask_{i+1}.png'), compress_level=0)
        with open(osp.join(save_path, f'{img_name}_info_{i+1}.json'), 'w', encoding='utf-8') as f:
            json.dump({"pos": pos}, f, ensure_ascii=False)

        dilate_in = osp.join(save_path, f'{img_name}_mask_{i+1}.png')
        dilate_out = osp.join(save_path, f'{img_name}_mask2_{i+1}.png')
        if erode_kernel_size > 0:
            erode(dilate_in, dilate_out,
                  erode_kernel_size, erode_iterations)
        else:
            shutil.copy2(dilate_in, dilate_out)


def denoise_mask(mask: torch.Tensor, opening_kernel_size: int = 5, closing_kernel_size: int = 5) -> torch.Tensor:
    """
    

     PyTorch Tensor NumPy  OpenCV
     PyTorch Tensor

    Args:
        mask (torch.Tensor):  PyTorch Tensor
                              (H, W, C)  (C, H, W)
                              0-1 uint8 0-255
        opening_kernel_size (int): 
                                   
        closing_kernel_size (int): 
                                   

    Returns:
        torch.Tensor: 
    """
    # 1. 
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f" torch.Tensor,  {type(mask)}")
    if mask.ndim != 3:
        raise ValueError(" Tensor  (H, W, C)  (C, H, W)")
    if opening_kernel_size <= 0 or opening_kernel_size % 2 == 0 or \
       closing_kernel_size <= 0 or closing_kernel_size % 2 == 0:
        raise ValueError("")

    # 2.  Torch Tensor  OpenCV  NumPy 
    # 
    original_device = mask.device
    original_dtype = mask.dtype
    original_shape = mask.shape

    #  (H, W, C) vs (C, H, W)
    channels_last = True
    if original_shape[0] in [1, 3] and original_shape[2] not in [1, 3]:
        channels_last = False  #  C, H, W 

    #  Tensor  (H, W, C)  NumPy 
    if channels_last:
        numpy_mask = mask.cpu().numpy()
    else:  # C, H, W -> H, W, C
        numpy_mask = mask.permute(1, 2, 0).cpu().numpy()

    #  uint8 [0, 255]
    if numpy_mask.dtype in [np.float32, np.float64, np.float16]:
        if np.max(numpy_mask) <= 1.0:
            numpy_mask = (numpy_mask * 255).astype(np.uint8)
        else:
            numpy_mask = numpy_mask.astype(np.uint8)
    elif numpy_mask.dtype != np.uint8:
        numpy_mask = numpy_mask.astype(np.uint8)

    # 3.  OpenCV 
    # 
    if numpy_mask.shape[2] == 3:
        gray_mask = cv2.cvtColor(numpy_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = np.squeeze(numpy_mask, axis=2)

    #  (0  255)
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # 
    opening_kernel = np.ones(
        (opening_kernel_size, opening_kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)

    # 
    closing_kernel = np.ones(
        (closing_kernel_size, closing_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, closing_kernel)

    # 4.  NumPy  Torch Tensor
    #  (H, W, C)
    if numpy_mask.shape[2] == 3:
        final_numpy_mask = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2BGR)
    else:
        final_numpy_mask = np.expand_dims(closed_mask, axis=2)

    #  NumPy  Tensor
    output_tensor = torch.from_numpy(final_numpy_mask)

    #  (H, W, C)  (C, H, W)
    if not channels_last:
        output_tensor = output_tensor.permute(2, 0, 1)

    # 
    if original_dtype.is_floating_point:
        output_tensor = output_tensor.to(original_dtype) / 255.0
    else:
        output_tensor = output_tensor.to(original_dtype)

    #  (CPU  GPU)
    output_tensor = output_tensor.to(original_device)

    return output_tensor


def denoise_mask_fast(
    mask: torch.Tensor, 
    min_area: int = 60, 
    filter_strips: bool = False,
    max_aspect_ratio: float = 5.0
) -> torch.Tensor:
    """
    
     PyTorch Tensor NumPy  OpenCV
     PyTorch Tensor
    
    args:
        mask (torch.Tensor):  PyTorch Tensor
        min_area (int): 
        filter_strips (bool): ;
        max_aspect_ratio (float):  filter_strips  True 
    """
    # --- 1.  () ---
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f" torch.Tensor,  {type(mask)}")
    
    original_device = mask.device
    original_dtype = mask.dtype
    original_shape = mask.shape

    # 
    channels_last = True
    if original_shape[0] in [1, 3] and original_shape[2] not in [1, 3]:
        channels_last = False 

    # Tensor -> Numpy (GPU -> CPU  Kornia/CuPy)
    if channels_last:
        numpy_mask = mask.cpu().numpy()
    else:
        numpy_mask = mask.permute(1, 2, 0).cpu().numpy()

    #  uint8
    if numpy_mask.dtype != np.uint8:
        if numpy_mask.dtype in [np.float32, np.float64, np.float16] and np.max(numpy_mask) <= 1.0:
            numpy_mask = (numpy_mask * 255).astype(np.uint8)
        else:
            numpy_mask = numpy_mask.astype(np.uint8)

    # 
    if numpy_mask.shape[2] == 3:
        gray_mask = cv2.cvtColor(numpy_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = np.squeeze(numpy_mask, axis=2)

    # 
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # --- 2.  ---
    
    # num_labels: 
    # labels: (H, W)  ID
    # stats: (N, 5) 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 
    if num_labels <= 1:
        final_mask = np.zeros_like(binary_mask)
    else:
        #  ()
        # stats 0:x, 1:y, 2:w, 3:h, 4:area
        areas = stats[:, cv2.CC_STAT_AREA]
        w = stats[:, cv2.CC_STAT_WIDTH]
        h = stats[:, cv2.CC_STAT_HEIGHT]

        #  A:  num_labels
        # keep_indices[i]  ID  i 
        keep_indices = np.ones(num_labels, dtype=bool)
        
        #  (ID=0) 
        keep_indices[0] = False

        #  B:  ()
        keep_indices &= (areas >= min_area)

        #  C:  ()
        if filter_strips:
            # 
            aspect_ratios = np.maximum(w, h) / (np.minimum(w, h) + 1e-5)
            
            # “”  
            is_strip = (aspect_ratios > max_aspect_ratio) & (areas < 5000)
            
            # 
            keep_indices &= (~is_strip)

        #  D:  -  NumPy “”
        # labels  (H, W) keep_indices  (N,) 
        # keep_indices[labels]  (H, W)  True 
        final_mask = (keep_indices[labels]).astype(np.uint8) * 255

    # --- 3.  () &  Tensor ---
    
    # 
    # closing_kernel = np.ones((3, 3), np.uint8)
    # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, closing_kernel)

    if numpy_mask.shape[2] == 3:
        final_numpy_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    else:
        final_numpy_mask = np.expand_dims(final_mask, axis=2)

    output_tensor = torch.from_numpy(final_numpy_mask)

    if not channels_last:
        output_tensor = output_tensor.permute(2, 0, 1)

    if original_dtype.is_floating_point:
        output_tensor = output_tensor.to(original_dtype) / 255.0
    else:
        output_tensor = output_tensor.to(original_dtype)

    return output_tensor.to(original_device)
