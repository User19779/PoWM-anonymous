from PIL import Image
import math
import os
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
import random


from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch

from typing import Tuple, List
from typing import Optional
import cv2

def get_gpu_free_memory(device:torch.device|str="cuda:0"):
    """
     GPU MB
    """
    if not torch.cuda.is_available():
        return 0
    
    #  (free_memory, total_memory) Byte
    free_byte, total_byte = torch.cuda.mem_get_info(device)
    
    #  MB
    free_mb = free_byte / (1024 ** 2)
    return free_mb


def show_images_side_by_side(images, titles=None, figsize=(8, 4)):
    """
    Display a list of images side by side.

    Args:
    images (list of numpy arrays): List of images to display.
    titles (list of str, optional): List of titles for each image. Default is None.
    """
    num_images = len(images)

    if titles is not None:
        if len(titles) != num_images:
            raise ValueError(
                "Number of titles must match the number of images.")

    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis('off')

        if titles is not None:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
    return


def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    with torch.no_grad():
        latents_pil_img = pipe.numpy_to_pil(
            pipe.decode_latents(latent.detach()))[0]
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    show_images_side_by_side([latents_pil_img, pil_img], [
                             'Latent', 'Generated Image'])
    return


def save_img(path: str | None, img_tensor: torch.Tensor):
    #  PyTorch 
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(" PyTorch ")
    if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.ndim != 3:
        raise ValueError(" [C, H, W]  [1, C, H, W]")

    if img_tensor.max() <= 1.05:
        img_tensor = img_tensor.mul(255).byte()
    #  CHW  HWC 
    img_tensor = img_tensor.permute(1, 2, 0).cpu()

    img_array = img_tensor.numpy()
    pil_image = Image.fromarray(img_array)
    if path:
        pil_image.save(path, compress_level=0)
    return pil_image


def _round_up_to_8(n):
    return (n + 7) // 8 * 8


# def _make_square(img: Image.Image, fill_color=(255, 255, 255)) -> Image.Image:
#     width, height = img.size
#     # 
#     max_size = _round_up_to_8(max(width, height))

#     # 
#     new_img = Image.new("RGB", (max_size, max_size), fill_color)
#     paste_position = ((max_size - width) // 2, (max_size - height) // 2)
#     new_img.paste(img, paste_position)
#     return new_img


def erode(infile: str, outfile: str, ksize: int = 3, iterations: int = 2):
    mask = cv2.imread(infile, 0)  # 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    cv2.imwrite(outfile, eroded_mask)


def adjust_size(img: Image.Image, output_size: tuple[int] | None = None, fill_color: str | int = 0) -> Image.Image:
    # )
    width, height = img.size
    if isinstance(output_size, tuple):
        target_width, target_height = output_size
    elif output_size == None:
        target_width = target_height = _round_up_to_8(max(width, height))

    # 
    new_img = Image.new(
        img.mode, (target_width, target_height), fill_color)  # "L"
    # paste_position = ((target_width - width) // 2,
    #                   (target_height - height) // 2)
    paste_position = (0, 0)
    
    new_img.paste(img, paste_position)
    new_img.crop((0, 0, target_width-1, target_height-1))

    return new_img


def mask_img_tensor(
        img_tensor, mask_path: str, mask_value=0, background_image: str | None = None) -> torch.Tensor | None:

    img_mask = Image.open(mask_path).convert("1")
    mask_width = img_mask.size[-2]
    mask_height = img_mask.size[-1]

    # maskimg_tensormaskimg_tensor
    if (mask_width != img_tensor.shape[-1] or mask_height != img_tensor.shape[-2]):
        img_mask = adjust_size(img_mask, tuple(img_tensor.shape[-1:-3:-1]), 0)
    img_mask_tensor = pil_to_tensor(img_mask).repeat((3, 1, 1))

    background_value = mask_value
    if bool(background_image):
        # 
        background = Image.open(background_image).convert("RGB")
        background = adjust_size(background, tuple(
            img_tensor.shape[-1:-3:-1]), "white")
        background_value_tensor = pil_to_tensor(background)
    else:
        if isinstance(background_value, int):
            background_value_tensor = torch.full_like(
                img_tensor, background_value,)
        else:
            background_value_tensor = torch.tensor(background_value)
            background_value_tensor = background_value_tensor.unsqueeze(
                -1).unsqueeze(-1)
            background_value_tensor = background_value_tensor.expand_as(
                img_tensor)

    img_tensor = torch.where(
        img_mask_tensor, img_tensor, background_value_tensor)
    return img_tensor


def get_img_tensor(
        img_path, device, mask: str | None = None, mask_value: int | tuple = 0,
        background_image: str | None = None,
        return_int: bool = False) -> torch.Tensor:
    """torch.Tensor0-1.

    Returns:
        torch.Tensor: 
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = pil_to_tensor(img)

    if bool(mask):
        img_tensor = mask_img_tensor(
            img_tensor, mask, mask_value, background_image)
    if return_int:
        pass
    else:
        img_tensor = (img_tensor/255)
    return img_tensor.to(device)


def create_output_folder(cfgs):
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])

    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join(
                [cfgs['case'][:-1], key+'/']))
            break

    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path

# Metrics for similarity


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.) - 10 * math.log10(mse)


def compute_psnr_outer_rectangle(a, b):
    """
     [C, H, W]  [N, C, H, W] 
     PSNR
    """
    #  [N, C, H, W] 
    if a.dim() == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    
    N, C, H, W = a.shape
    psnr_list = []

    for i in range(N):
        img_a = a[i] # [C, H, W]
        img_b = b[i] # [C, H, W]

        # 1.  ( any)
        # mask : [H, W]
        mask = (img_b > 0).any(dim=0)
        coords = torch.nonzero(mask)

        if coords.shape[0] == 0:
            psnr_list.append(100.0)
            continue

        # 2. 
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        # 3. 
        roi_a = img_a[:, y_min:y_max+1, x_min:x_max+1]
        roi_b = img_b[:, y_min:y_max+1, x_min:x_max+1]

        # 4.  MSE
        mse = torch.mean((roi_a - roi_b) ** 2).item()

        if mse == 0:
            psnr_list.append(100.0)
        else:
            #  [0, 1]
            psnr = -10 * math.log10(mse)
            psnr_list.append(psnr)

    #  Batch 
    return sum(psnr_list) / len(psnr_list)


def ssim_outer_rectangle(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    SSIM 
     X  Y  SSIM
    X, Y  [B, C, H, W]
    """
    if not X.shape == Y.shape:
        raise ValueError(f" {X.shape}  {Y.shape}")

    # 1.  (Mask)
    #  1 (N, C, H, W) > 0 
    # mask  (N, H, W)
    mask = (X > 0.02).any(dim=1) | (Y > 0.02).any(dim=1)

    # 2.  Batch 
    # coords  (TotalNonZeroPixels, 3) [batch_idx, y, x]
    coords = torch.nonzero(mask)

    # (y)(x)
    y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
    y_max, x_max = coords[:, 1].max(), coords[:, 2].max()

    # 3. 
    #  Batch 
    X_crop = X[:, :, y_min:y_max+1, x_min:x_max+1]
    Y_crop = Y[:, :, y_min:y_max+1, x_min:x_max+1]

    h_crop, w_crop = X_crop.shape[-2:]
    ans = compute_ssim(X_crop, Y_crop,)
    return ans


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()


def load_img(img_path, device):
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x


def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(
        new_img_path, device)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)


def eval_lpips(ori_img_path, new_img_path, metric, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(
        new_img_path, device)
    return metric(ori_x, new_x).item()

# Detect watermark from one image


def watermark_prob(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda')):
    if isinstance(img, str):
        img_tensor = get_img_tensor(img, device=device,)
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    det_prob = wm_pipe.one_minus_p_value(
        reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    return det_prob


def img_tensor_to_numpy(tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()


def shift_image_tensor(image_tensor: torch.Tensor, direction: str, pixels_to_shift: int) -> torch.Tensor:
    """
    Shifts an image tensor (C, H, W) or (N, C, H, W) in a specified direction
    by a given number of pixels, filling the newly created blank areas with black pixels (zeros).

    Args:
        image_tensor (torch.Tensor): The input image tensor.
                                     Expected shape: (C, H, W) for a single image,
                                     or (N, C, H, W) for a batch of images.
                                     Pixel values are assumed to be between 0 and 255 (or normalized).
        direction (str): The direction to shift the image. Must be one of
                         'left', 'right', 'up', or 'down'.
        pixels_to_shift (int): The number of pixels to shift the image.
                               Must be a non-negative integer. If it's greater than
                               the image dimension in that direction, the image
                               will become entirely black.

    Returns:
        torch.Tensor: The shifted image tensor with the same shape as the input,
                      with new areas filled with black pixels (zeros).

    Raises:
        ValueError: If the direction is invalid or pixels_to_shift is negative.
    """
    if direction not in ['left', 'right', 'up', 'down']:
        raise ValueError(
            "Invalid direction. Must be 'left', 'right', 'up', or 'down'.")
    if pixels_to_shift < 0:
        raise ValueError("pixels_to_shift must be a non-negative integer.")

    # Determine if it's a batch of images or a single image
    is_batch = len(image_tensor.shape) == 4
    if is_batch:
        num_batches, channels, height, width = image_tensor.shape
    else:
        channels, height, width = image_tensor.shape
        # Add a batch dimension for consistent processing
        image_tensor = image_tensor.unsqueeze(0)
        num_batches = 1

    # Create a new tensor filled with zeros (black pixels) of the same shape
    shifted_image_tensor = torch.zeros_like(image_tensor)

    # Apply the shift based on the direction
    if direction == 'left':
        # Copy the right part of the original image to the left part of the new tensor
        # The first 'pixels_to_shift' columns on the right will be black
        shifted_image_tensor[:, :, :, :-
                             pixels_to_shift] = image_tensor[:, :, :, pixels_to_shift:]
    elif direction == 'right':
        # Copy the left part of the original image to the right part of the new tensor
        # The first 'pixels_to_shift' columns on the left will be black
        shifted_image_tensor[:, :, :,
                             pixels_to_shift:] = image_tensor[:, :, :, :-pixels_to_shift]
    elif direction == 'up':
        # Copy the bottom part of the original image to the top part of the new tensor
        # The first 'pixels_to_shift' rows at the bottom will be black
        shifted_image_tensor[:, :, :-pixels_to_shift,
                             :] = image_tensor[:, :, pixels_to_shift:, :]
    elif direction == 'down':
        # Copy the top part of the original image to the bottom part of the new tensor
        # The first 'pixels_to_shift' rows at the top will be black
        shifted_image_tensor[:, :, pixels_to_shift:,
                             :] = image_tensor[:, :, :-pixels_to_shift, :]

    # Remove the batch dimension if the input was a single image
    if not is_batch:
        shifted_image_tensor = shifted_image_tensor.squeeze(0)

    return shifted_image_tensor


def occlude_random_block(
    image_tensor: torch.Tensor,
    M: int,
    direction: str
) -> torch.Tensor:
    """
    Randomly blocks continuous M rows or columns of a PyTorch image tensor,
    filling the area with black pixels (value 0).

    Args:
        image_tensor (torch.Tensor): The input image tensor. Expected format is
                                     (C, H, W) for color images or (H, W) for
                                     grayscale images.
        M (int): The number of continuous rows or columns to block. Must be
                 a positive integer.
        direction (str): The direction to block. Must be 'rows' or 'columns'.

    Returns:
        torch.Tensor: A new tensor with the random block occluded. The
                      original tensor is not modified.

    Raises:
        ValueError: If `direction` is not 'rows' or 'columns'.
        ValueError: If `M` is not a positive integer or is larger than the
                    corresponding dimension of the image.
    """
    # Ensure the direction is valid
    if direction not in ['rows', 'columns']:
        raise ValueError("Direction must be 'rows' or 'columns'.")

    # Get the height and width of the image tensor.
    # We use -2 and -1 to handle both (H, W) and (C, H, W) formats.
    H, W = image_tensor.shape[-2:]

    # Validate M against the image dimensions
    if M <= 0 or not isinstance(M, int):
        raise ValueError("M must be a positive integer.")
    if direction == 'rows' and M > H:
        raise ValueError(
            f"M ({M}) cannot be greater than image height ({H}) for 'rows' direction.")
    if direction == 'columns' and M > W:
        raise ValueError(
            f"M ({M}) cannot be greater than image width ({W}) for 'columns' direction.")

    # modifying the original to keep gradient flow
    output_tensor = image_tensor

    if direction == 'rows':
        # Select a random starting row index for the block.
        # The range is from 0 to H - M, inclusive, to ensure the block fits.
        start_row = torch.randint(0, H - M + 1, (1,)).item()

        # Occlude the selected rows with black pixels (value 0)
        # The '...' slices across all leading dimensions (like the channel)
        # to apply the change to all channels.
        output_tensor[..., start_row: start_row + M, :] = 0
    else:  # direction == 'columns'
        # Select a random starting column index for the block.
        start_col = torch.randint(0, W - M + 1, (1,)).item()

        # Occlude the selected columns with black pixels
        output_tensor[..., :, start_col: start_col + M] = 0

    return output_tensor
