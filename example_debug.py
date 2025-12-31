# %%
# 
from main.attdiffusion import ReSDPipeline  # 
from main.divide_picture import save_adjust_image, GetDivideMethod, denoise_mask_fast  # 
from main.wmattacker import *  # 
from main.utils import *  # 

import torchvision.transforms as transforms
from torchvision.utils import save_image
from transformers import AutoImageProcessor, BitImageProcessor, AutoModel


# 
from loss.pytorch_ssim import ssim  # SSIM
from loss.loss import LossProvider  # 

# Diffusers
import diffusers
from diffusers.utils.torch_utils import randn_tensor  # 
from diffusers import DDIMScheduler, StableDiffusionPipeline  # 

# 
from datasets import load_dataset  # 
import torchvision.transforms as transforms  # 
from PIL import Image  # 
import rawpy  # RAW
import imageio  # 

# PyTorch
import torch.optim as optim  # 
import torch  # PyTorch
import torch.nn as nn
import torch.nn.functional as F

# 
import argparse  # 
import yaml  # YAML
import os  # 
import logging  # 
import shutil  # 
import numpy as np  # 
import json  # JSON
import gc  # 
import os.path as osp  # 
import csv  # CSV
import time

# 
import rich  # 
import rich.progress

import torch
from torchvision.transforms import InterpolationMode

from main.processor import BitImageProcessor
# 
from main.position_feature_INN import position_feature_INN, PositionSematicLoss
from loss.edge_compute import sobel_edge_detection

import os
os.environ["http_proxy"]="http://172.32.52.144:12798"
os.environ["https_proxy"]="http://172.32.52.144:12798"

# =================================================================================
# 1.  (Argument Parsing)
#    
# =================================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # ---  ---
    parser.add_argument('--device', type=str, default='cuda:0',
                        help=',  "cuda:0"  "cpu"')
    parser.add_argument('--config_path', type=str,
                        default='./example/config/config_b.yaml', help='YAML')
    parser.add_argument('--json_file_path', type=str,
                        required=True, help='JSON')
    parser.add_argument('--image_folder', type=str,
                        required=True, help='')
    parser.add_argument('--mask_folder', type=str,
                        required=True, help='')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results', help='')

    # ---  ---
    parser.add_argument('--A', type=int, default=5, help='')
    parser.add_argument('--erode_kernel_size', type=int,
                        default=4, help='')
    parser.add_argument('--erode_iterations', type=int,
                        default=1, help='')
    parser.add_argument('--special_pos_num', type=int,
                        default=2, help='single_picture_only')
    parser.add_argument('--single_picture_only',
                        action='store_true', help='')

    # ---  ---
    parser.add_argument('--message_bits', type=int,
                        default=32, help='')
    parser.add_argument('--L', type=int, default=128, help='')
    parser.add_argument('--miu_value', type=float,
                        default=0.3, help='margin (μ)')
    parser.add_argument('--block_background',
                        action='store_true', help='')
    parser.add_argument('--block_background_b',
                        action='store_true', help='')
    parser.add_argument('--noise_1_std', type=float,
                        default=0.25, help='')
    parser.add_argument('--noise_2_std', type=float,
                        default=0.10, help='')
    parser.add_argument('--learning_rate', type=float,
                        default=0.02, help='')

    parser.add_argument('--loss_weights',
                        type=float,      # 
                        nargs=3,         #  3 !
                        default=[0.25, 0.25, 0.50],
                        metavar=('Weight1', 'Weight2', 'Weight3'),
                        help='')
    
    parser.add_argument('--start_index', type=int, default=0,
                        help=' ()')
    parser.add_argument('--end_index', type=int, default=-1,
                        help=' (, -1 )')

    parser.add_argument('--save_iters', type=str,
                        help='')

    return parser.parse_args()


args = parse_args()

# =================================================================================
# 2.  (Global Constants & Environment Setup)
# =================================================================================
# ---  ---
DEVICE = torch.device(args.device)
CONFIG_PATH = args.config_path
JSON_FILE_PATH = args.json_file_path
IMAGE_FOLDER = args.image_folder
MASK_FOLDER = args.mask_folder
OUTPUT_DIR = args.output_dir

A = args.A
ERODE_KERNEL_SIZE = args.erode_kernel_size
ERODE_INTERACTIONS = args.erode_iterations
SPECIAL_POS_NUM = args.special_pos_num
SINGLE_PICTURE_ONLY = args.single_picture_only

MESSAGE_BITS = args.message_bits
L = args.L
MIU_VALUE = args.miu_value
BLOCK_BACKGROUND = args.block_background
BLOCK_BACKGROUND_B = args.block_background_b
NOISE_1_STD = args.noise_1_std
NOISE_2_STD = args.noise_2_std

LOSS_WEIGHTS = args.loss_weights
LEARNING_RATE = args.learning_rate

SAVE_ITERS = args.save_iters


CSV_RESULT_PATH = osp.join(OUTPUT_DIR, "results.csv")
CSV_QUALITY_PATH = osp.join(
    OUTPUT_DIR, "debug_csv_3.csv")  # <--- 1CSV


# ---  ---
WM_PATH = osp.join(OUTPUT_DIR, 'watermarked')  # 
AFTER_MASK_FOLDER = osp.join(OUTPUT_DIR, 'intermediate')  # 
LOG_FILE_NAME = osp.join(OUTPUT_DIR, "debug_log")  # 
os.makedirs(WM_PATH, exist_ok=True)
os.makedirs(AFTER_MASK_FOLDER, exist_ok=True)

# ---  ---
logger = logging.getLogger('terminal_logger')
logger.setLevel(logging.INFO)
file_handler_1 = logging.FileHandler(f'{LOG_FILE_NAME}_1.log')
file_handler_1.setLevel(logging.INFO)
logger.addHandler(file_handler_1)


# --- CSV ---
attack_folders_for_header = ['Original', 'diff_attacker_60', 'diff_attacker_30', 'cheng200-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50',
                             'brightness_0.5', 'contrast_0.8', 'contrast_1.2', 'Gaussian_noise', 'Gaussian_blur',
                              'all_norot', 'No']
csv_header = ['Image'] + attack_folders_for_header
with open(CSV_RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

# <--- 2CSV ---
quality_csv_header = ['Image', 'SSIM', 'PSNR']
with open(CSV_QUALITY_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(quality_csv_header)
# --- ---

os.environ['TORCH_HOME'] = './torch_cache'

# %% [markdown]
# ## Necessary Setup for All Sections

# %%
logger.info(f'===== Load Config =====')
logger.info(f'Running on device: {DEVICE}')
device = DEVICE

with open(CONFIG_PATH, 'r') as file:
    cfgs = yaml.safe_load(file)
# 
cfgs['save_img'] = WM_PATH
cfgs['loss_weights'] = LOSS_WEIGHTS

if isinstance(SAVE_ITERS, str) and len(SAVE_ITERS) > 0:
    save_iters_str = SAVE_ITERS.split(" ")
    cfgs['save_iters'] = [int(i) for i in save_iters_str]
    cfgs['iters'] = cfgs['save_iters'][-1]
    print(f" {save_iters_str} epoch")


logger.info(cfgs)


# VAE
model_path = "./stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    os.path.abspath(model_path), torch_dtype=torch.float16)
pipe.vae.to(device=device)

# image encoder
#  DINO v2 Small 
model_name = "./facebook-dinov2-small"
processor = BitImageProcessor()
processor.do_rescale = False

model = AutoModel.from_pretrained(os.path.abspath(model_name),)
model.to(device)

# (:  A = 5  args.A )

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    imagename_list_original = json.load(f)

# (:  IMAGE_FOLDER, MASK_FOLDER, AFTER_MASK_FOLDER, ERODE_KERNEL_SIZE, ERODE_INTERACTIONS )

imagename_list = imagename_list_original

# <---  --->
# 
START_INDEX = args.start_index
END_INDEX = args.end_index

total_valid_images = len(imagename_list)
real_end_index = total_valid_images if (END_INDEX == -1 or END_INDEX > total_valid_images) else END_INDEX

# 
logger.info(f"Total valid images found: {total_valid_images}")
logger.info(f"Processing range: [{START_INDEX}:{real_end_index}]")

# 
imagename_list = imagename_list[START_INDEX : real_end_index]

if len(imagename_list) == 0:
    logger.warning("No images to process in the specified range!")
# <---  --->

if True: 
    for imagename in rich.progress.track(imagename_list):
        gt_img_tensor = get_img_tensor(
            osp.join(IMAGE_FOLDER, f'{imagename}.png'), device,
            mask=osp.join(MASK_FOLDER, f'{imagename}.png'), mask_value=(0, 0, 0))

        if True:
            image_after_mask = transforms.ToPILImage("RGB")(gt_img_tensor)
            image_after_mask.save(
                f'{AFTER_MASK_FOLDER}/{imagename}.png', compress_level=0)
        gt_img_tensor = gt_img_tensor.unsqueeze(0)

        gt_mask = Image.open(f'{MASK_FOLDER}/{imagename}.png').convert("1")
        # gt_mask = adjust_size(gt_mask,
        #     (int(gt_img_tensor.shape[-2]), int(gt_img_tensor.shape[-1])))

        gt_mask_tensor = pil_to_tensor(gt_mask)
        pos = GetDivideMethod(gt_mask_tensor)

        with open(f'{MASK_FOLDER}/{imagename}_info.json', 'w', encoding='utf-8') as f:
            json.dump({"pos": pos}, f, ensure_ascii=False)

        save_adjust_image(
            image_after_mask, gt_mask, pos,
            AFTER_MASK_FOLDER, imagename,
            background_color=(0, 0, 0), A=A,
            erode_kernel_size=ERODE_KERNEL_SIZE, erode_iterations=ERODE_INTERACTIONS)

        del gt_img_tensor, image_after_mask, gt_mask, gt_mask_tensor,


for imagename in imagename_list:
    # memory_avaliable = get_gpu_free_memory(device)
    # while(memory_avaliable<20000):
    #     time.sleep(10)
    #     memory_avaliable = get_gpu_free_memory(device)
    
    gt_img_tensors: list[torch.Tensor] = list()
    mask_tensors_not_denoised: list[torch.Tensor] = list()
    mask2_tensors_not_denoised: list[torch.Tensor] = list()
    for pos_num in range(A):
        gt_img_tensors.append(get_img_tensor(
            f'{AFTER_MASK_FOLDER}/{imagename}_pos_{pos_num+1}.png', device,
            mask=f'{AFTER_MASK_FOLDER}/{imagename}_mask_{pos_num+1}.png', mask_value=(0, 0, 0),),)

        mask_tensors_not_denoised.append(get_img_tensor(
            f'{AFTER_MASK_FOLDER}/{imagename}_mask_{pos_num+1}.png', device,))
        mask2_tensors_not_denoised.append(get_img_tensor(
            f'{AFTER_MASK_FOLDER}/{imagename}_mask2_{pos_num+1}.png', device,))

    mask_tensors: list[torch.Tensor] = [
        denoise_mask_fast(i) for i in mask_tensors_not_denoised]
    mask2_tensors: list[torch.Tensor] = [
        denoise_mask_fast(i) for i in mask2_tensors_not_denoised]

    watermark_shape = (
        1, 4, gt_img_tensors[0].shape[-2]//8, gt_img_tensors[0].shape[-1]//8)
    rich.print(f"Watermark Shape:[{watermark_shape}]")
    for pos_num in range(A):
        gt_img_tensors[pos_num] = torch.unsqueeze(gt_img_tensors[pos_num], 0)
        mask_tensors[pos_num] = torch.unsqueeze(mask_tensors[pos_num], 0)

    wm_path = cfgs['save_img']  # 
    logger.info(f'===== Init Pipeline  {imagename}=====')
    pipe.to(device)
    model.to(device)

    torch.serialization.add_safe_globals([position_feature_INN])
    torch.serialization.add_safe_globals([set])

    # %% [markdown]
    # ## Image Watermarking

    # %%
    # Step 1: Get init noise

    def get_init_latent(img_tensor: torch.Tensor, pipe):
        img_tensor_fp16 = img_tensor.to(torch.float16)
        img_tensor_normalized = 2 * img_tensor_fp16 - 1.0
        ans_img_latent = pipe.vae.encode(
            img_tensor_normalized).latent_dist.mode()
        ans_img_latent = ans_img_latent.to(img_tensor.dtype)
        ans_img_latent = ans_img_latent * pipe.vae.config.scaling_factor
        return ans_img_latent

    def revert_init_latent(img_latent: torch.Tensor, pipe):
        img_latent_fp16 = img_latent.to(torch.float16)
        img_latent_fp16 = img_latent_fp16 / pipe.vae.config.scaling_factor
        decoded_images = pipe.vae.decode(img_latent_fp16).sample
        decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        decoded_images = decoded_images.to(img_latent.dtype)
        return decoded_images

    def get_modified_latent(img_latent, delta_tensor, mode='spatial_to_freq'):
        """
        img_latent: 
        delta_tensor: 
        mode: 
            'freq_add': delta
            'spatial_to_freq': deltaFFT
        """
        # 1.  FFT
        img_latent_fft = torch.fft.fftshift(
            torch.fft.fft2(img_latent.to(torch.float32)), dim=(-1, -2)
        )
        
        if mode == 'freq_add':
            # 
            img_latent_fft = img_latent_fft + delta_tensor
        elif mode == 'spatial_to_freq':
            delta_fft = torch.fft.fftshift(
                torch.fft.fft2(delta_tensor.to(torch.float32)), dim=(-1, -2)
            )
            img_latent_fft = img_latent_fft + delta_fft

        # 2. IFFT 
        ans_img_latent = torch.fft.ifft2(
            torch.fft.ifftshift(img_latent_fft, dim=(-1, -2))
        ).real
        
        ans_img_latent = ans_img_latent.to(img_latent.dtype)
        return ans_img_latent


    def get_modified_latent_spacial(
        img_tensor, delta_img_tensor, sematic_network: position_feature_INN | None = None):
        img_tensor_f32 = img_tensor.to(torch.float32)
        # 2. 
        ans_img_tensor = img_tensor_f32 + delta_img_tensor.to(torch.float32)
        ans_img_tensor = ans_img_tensor.to(img_tensor.dtype)
        return ans_img_tensor
    
    
    def total_variation_loss(img_tensor):
        """
        
        """
        b, c, h, w = img_tensor.size()
        tv_h = torch.pow(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)


    # %%
    # Step 2: prepare training
    init_latents = list()

    # (:  single_picture_only  SPECIAL_POS_NUM )
    single_picture_only = SINGLE_PICTURE_ONLY

    if single_picture_only:
        init_latents_approx = get_init_latent(
            gt_img_tensors[SPECIAL_POS_NUM], pipe)
        init_latents = [torch.zeros_like(
            init_latents_approx, device=device) for j in range(A)]
        init_latents[SPECIAL_POS_NUM] = (init_latents_approx.detach().clone())
        init_latents[SPECIAL_POS_NUM].requires_grad = True
        del init_latents_approx
    else:
        for pos_num in range(A):
            init_latents_approx = get_init_latent(
                gt_img_tensors[pos_num], pipe)
            init_latents.append(init_latents_approx.detach().clone())
            init_latents[pos_num].requires_grad = True
        del init_latents_approx

    delta_latent = torch.zeros_like(
        init_latents[0], dtype=torch.float32, device=device)
    delta_latent.normal_(mean=0, std=0.001)
    delta_latent.requires_grad = True
    


    optimizer = optim.Adam([
        {'params': [delta_latent], 'lr': LEARNING_RATE},
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200], gamma=1.0)
    

    
    # (: MESSAGE_BITS, L )
    info_to_mask = torch.tensor(
        [0, 0, 1, 1], dtype=int, device=device).repeat(MESSAGE_BITS//4)
    identity_matrix = torch.eye(384, device=device)
    fixed_direction_vectors = identity_matrix[:MESSAGE_BITS, :L]
    fixed_direction_vectors.requires_grad = True
    train_target_sims = torch.where(
        info_to_mask == 1, 1, -1).to(torch.float)

    totalLoss = LossProvider(
        cfgs['loss_weights'], device,)
    loss_lst = []

    # %%
    # Step 3: train the init latents·
    # (: BLOCK_BACKGROUND, BLOCK_BACKGROUND_B, NOISE_1_STD, NOISE_2_STD )
    for iter in range(cfgs['iters']):
        do_log_flag = (iter % 10 == 9)
        if do_log_flag:
            logger.info(f'iter {iter}:')

        pred_img_tensors = []
        optimizer.zero_grad()

        INIT_ITERS = 0
        for pos_num in range(0, A):
            if single_picture_only and pos_num != SPECIAL_POS_NUM:
                continue
            elif (pos_num == SPECIAL_POS_NUM) or (iter < INIT_ITERS):
                noise_variant_iterator = range(4)
            else:
                noise_variant_iterator = (0,)

            for noise_variant in noise_variant_iterator:
                init_latent_watermarked_o = get_modified_latent(
                    init_latents[pos_num], delta_latent, mode='spatial_to_freq')

                pos_loss = torch.tensor(data=[0.0,], device=device)
                
                #  noise_1 
                # random_mean_1 = torch.empty(
                #     1, device=device).uniform_(-NOISE_1_STD, NOISE_1_STD).item()
                noise_1 = torch.normal(
                    mean=0.0, std=NOISE_1_STD,
                    size=init_latent_watermarked_o.shape, device=device
                )

                #  noise_2 
                # random_mean_2 = torch.empty(
                #     1, device=device).uniform_(-NOISE_2_STD, NOISE_2_STD).item()
                noise_2 = torch.normal(
                    mean=0.0, std=NOISE_2_STD,
                    size=gt_img_tensors[0].shape, device=device
                )

                if noise_variant == 1:
                    init_latent_watermarked = init_latent_watermarked_o + noise_1
                else:
                    init_latent_watermarked = init_latent_watermarked_o

                pred_img_tensor = revert_init_latent(
                    init_latent_watermarked, pipe)

                #   =  + 
                if torch.isnan(pred_img_tensor,).any():
                    raise RuntimeError("nan")

                if BLOCK_BACKGROUND:
                    mask_to_use = mask_tensors[pos_num]
                    img_tensor_for_total_loss = torch.where(
                        mask_to_use == 0,
                        torch.zeros_like(pred_img_tensor, device=device),
                        pred_img_tensor)
                else:
                    img_tensor_for_total_loss = pred_img_tensor
                    
                if noise_variant == 0:
                    pos_loss = pos_loss + totalLoss.forward(
                        img_tensor_for_total_loss, gt_img_tensors[pos_num],
                        mask_tensors[pos_num], do_log=do_log_flag)

                if noise_variant == 2:
                    pred_img_tensor = pred_img_tensor + noise_2
                    pred_img_tensor = torch.clamp(
                        pred_img_tensor, 0.0, 1.0)

                elif noise_variant == 3:
                    shift_direction: str = random.choice(
                        ("left", "right", "up", "down"))
                    shift_distance: int = random.randrange(2, 10, 1)
                    pred_img_tensor = shift_image_tensor(
                        pred_img_tensor, shift_direction, shift_distance
                    )

                if BLOCK_BACKGROUND_B:
                    mask_to_use = mask2_tensors[pos_num]
                    pred_img_tensor = torch.where(
                        mask_to_use == 0,
                        torch.zeros_like(pred_img_tensor, device=device),
                        pred_img_tensor)
                    del mask_to_use

                if iter == 0 or iter == 99:
                    inputs = processor.process(pred_img_tensor)
                    outputs = model(inputs)
                else:
                    inputs = processor.process(pred_img_tensor)
                    outputs = model(inputs)
                output_vector = outputs.last_hidden_state[0, 0, :L]
                del outputs

                # (: MIU_VALUE )
                output_score = torch.matmul(
                    output_vector, fixed_direction_vectors.T)

                output_score_b = MIU_VALUE - train_target_sims * output_score
                output_score_b = torch.clamp_min(output_score_b, 0.0)
                watermark_loss = torch.sum((output_score_b)) / MESSAGE_BITS

                info_revealed = torch.greater(output_score, 0.0).int()
                error_bits = (info_revealed != info_to_mask).sum()
                error_rate = error_bits / MESSAGE_BITS

                pos_loss = pos_loss + watermark_loss
                if do_log_flag:
                    logger.info(f'wt{noise_variant}_loss {watermark_loss:.5f}, bit err% {error_rate:.3f}')

                if not (single_picture_only and (pos_num != SPECIAL_POS_NUM)):
                    pos_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if ((iter+1) in cfgs['save_iters']):
            with torch.no_grad():
                pred_img_tensors: list[torch.Tensor] = []
                for pos_num in range(A):
                    if single_picture_only and pos_num != SPECIAL_POS_NUM:
                        continue
                    init_latent_watermarked = get_modified_latent(
                        init_latents[pos_num], delta_latent)
                    pred_img_tensor_rough = revert_init_latent(
                        init_latent_watermarked, pipe)
                    if BLOCK_BACKGROUND:
                        mask_to_use = mask_tensors[pos_num]
                        pred_img_tensor = torch.where(
                            mask_to_use == 0,
                            torch.zeros_like(pred_img_tensor, device=device),
                            pred_img_tensor_rough)
                    else:
                        pred_img_tensor = pred_img_tensor_rough
                        
                    pred_img_tensor = torch.clamp(
                        torch.nan_to_num(
                            pred_img_tensor, 0.5, 1.0, 0.0), 0.0, 1.0
                    )

                    path = os.path.join(
                        wm_path, f"{imagename.split('.')[0]}_{iter+1}_pos_{pos_num+1}.png")
                    save_img(path, pred_img_tensor)
                    pred_img_tensors.append(pred_img_tensor)
        scheduler.step()
    gc.collect()

    # Stage 2 
    STAGE_2_ITERS = 50
    #  Stage 1  0.05 
    diff_map = torch.abs(pred_img_tensor - gt_img_tensors[SPECIAL_POS_NUM])
    diff_map = diff_map 
    
    def get_continuous_artifact_mask_multichannel(
        pred_img, gt_img, 
        diff_threshold=0.04,  
        density_kernel=9,     
        density_threshold=0.4 
    ):
        """
        
        """
        # 1.  ( B, C, H, W)
        #  dim=1  mean
        diff_map = torch.abs(pred_img - gt_img)
        raw_mask = (diff_map > diff_threshold).float()

        # 2.  (Density Map)
        # F.avg_pool2d 
        padding = density_kernel // 2
        density_map = F.avg_pool2d(
            raw_mask, 
            kernel_size=density_kernel, 
            stride=1, 
            padding=padding
        )

        # 3. 
        #  block_mask  [B, 3, H, W]
        block_mask = (density_map > density_threshold).float()
        
        return block_mask

    def tv_loss(y):
        return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

    artifact_mask = get_continuous_artifact_mask_multichannel(
        pred_img_tensor*mask2_tensors[SPECIAL_POS_NUM],
        gt_img_tensors[SPECIAL_POS_NUM]*mask2_tensors[SPECIAL_POS_NUM])
        
    # save_image(artifact_mask,"M.png")
    original_rectangle = gt_img_tensors[SPECIAL_POS_NUM].detach()
    #  Stage 1 
    delta_2 = (torch.zeros_like(gt_img_tensors[0], device=device))
    # delta_2 = (torch.randn_like(gt_img_tensors[0], device=device))*5e-3
    delta_2.requires_grad = True
    optimizer_2 = optim.Adam([
        {'params': [delta_2], 'lr': 2e-2},
    ])
    
    with torch.no_grad():
        #  Stage 1 
        #  pred_img_tensor 
        anchor_inputs = processor.process(pred_img_tensor)
        anchor_outputs = model(anchor_inputs)
        stage1_feature_anchor = anchor_outputs.last_hidden_state[0, 0, :L].detach()
        del anchor_outputs, anchor_inputs
        
    loss_img = torch.sum(artifact_mask * torch.abs(pred_img_tensor - original_rectangle)) 
    loss_img = loss_img / (torch.sum(artifact_mask) + 1e-6)
    logger.info(f"\n\nBefore Finetune Img_Loss={loss_img.item():.4f}")   
     
    for i in range(STAGE_2_ITERS):
        optimizer_2.zero_grad()
        
        # ---  ---
        current_final_img = pred_img_tensor + delta_2 * artifact_mask
        input_for_detection = torch.clamp(current_final_img, 0.0, 1.0)
        current_final_img = current_final_img * mask_tensors[SPECIAL_POS_NUM]

        # ---  ---
        inputs = processor.process(input_for_detection)
        outputs = model(inputs)
        current_feature_vector = outputs.last_hidden_state[0, 0, :L]
        
        # ---  Loss ---
        # 2.  Loss ( Stage 1 )
        #  MSE || phi(I_final) - phi(I_stage1) ||^2
        loss_feat = F.mse_loss(current_feature_vector, stage1_feature_anchor)

        # 3.  Loss ()
        #  artifact 
        loss_img = torch.sum(artifact_mask * torch.abs(current_final_img - original_rectangle)) 
        loss_img = 3 * loss_img / (torch.sum(artifact_mask) + 1e-6)
        psnr_value = compute_psnr_outer_rectangle(current_final_img, original_rectangle)
        # 5.  (L1)
        loss_reg = torch.mean(torch.abs(delta_2))

        # ---  ---
        # loss_feat 
        total_loss_2 = 0.1 * loss_feat + loss_img
        
        total_loss_2.backward()
        optimizer_2.step()

        # ---  (Projected Gradient Descent) ---
        #  0.05  Stage 1 
        with torch.no_grad():
            delta_2.clamp_(-0.1, 0.1)
        
        if i % 5 == 0:
            save_img(f"{i}B.png", input_for_detection)
            logger.info(f"Stage 2 Iter {i}")
            logger.info(f"WM_Loss={loss_feat.item():.4f}, Img_Loss={loss_img.item():.4f},Reg_Loss={loss_reg.item():.4f}")
            logger.info(f"psnr_value {psnr_value:.5f}")
    
    path = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_pos_{SPECIAL_POS_NUM+1}.png")
    save_img(path, current_final_img.detach().cpu())

    def binary_search_theta():
        return 0

    # %%

    # 
    mask_tensor = get_img_tensor(
        osp.join(MASK_FOLDER, f'{imagename}.png'), device, return_int=True)
    gt_img_tensor = get_img_tensor(
        osp.join(IMAGE_FOLDER, f'{imagename}.png'), device,)
    with open(osp.join(MASK_FOLDER, f'{imagename}_info.json'), 'r', encoding='utf-8') as f:
        obj = json.load(f)
        pos = obj["pos"]
        min_x, min_y, max_x, max_y = pos
    original_rectangle = gt_img_tensor[
        :, min_y:max_y+1, min_x:max_x+1].detach().clone()

    wm_img_path = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_pos_{SPECIAL_POS_NUM+1}.png")
    wm_img_tensor = get_img_tensor(wm_img_path, device)

    wm_mask_path = f'{AFTER_MASK_FOLDER}/{imagename}_mask_{SPECIAL_POS_NUM+1}.png'
    wm_mask_tensor = get_img_tensor(wm_mask_path, device, return_int=True)
    with open(f'{AFTER_MASK_FOLDER}/{imagename}_info_{SPECIAL_POS_NUM+1}.json', 'r', encoding='utf-8') as f:
        obj = json.load(f)
        pos = obj["pos"]
        pos_min_x, pos_min_y, pos_max_x, pos_max_y = pos
    valid_rectangle_watermarked = wm_img_tensor[:,
                                                pos_min_y:pos_max_y+1, pos_min_x:pos_max_x+1].clone().detach()
    valid_rectangle_mask = wm_mask_tensor[:, pos_min_y:pos_max_y +
                                            1, pos_min_x:pos_max_x+1].clone().detach()

    print((min_x, min_y, max_x, max_y))
    print((pos_min_x, pos_min_y, pos_max_x, pos_max_y))

    valid_rectangle = torch.where(
        valid_rectangle_mask == 255, valid_rectangle_watermarked, original_rectangle
    )
    edited_img_tensor = gt_img_tensor.clone().detach()
    edited_img_tensor[:, min_y:max_y+1, min_x:max_x+1] = valid_rectangle

    ssim_value = ssim_outer_rectangle(
        valid_rectangle.unsqueeze(0),
        original_rectangle.unsqueeze(0))
    psnr_value=compute_psnr_outer_rectangle(valid_rectangle.unsqueeze(0), original_rectangle.unsqueeze(0))
    logger.info(f'Original SSIM {ssim_value}')

    optimal_theta = 0.0
    logger.info(f'Optimal Theta {optimal_theta}')
    if optimal_theta < 1e-2:
        img_tensor = edited_img_tensor
    else:
        img_tensor = (gt_img_tensor-edited_img_tensor) * \
            optimal_theta + edited_img_tensor
            
    # 
    ssim_threshold = 0.0
    image_after_mask_name = f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}"

    path = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")
    save_img(path, img_tensor)

    # <--- 3CSV ---
    quality_row_data = [imagename, f"{ssim_value:.5f}", f"{psnr_value:.5f}"]
    with open(CSV_QUALITY_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(quality_row_data)
    # --- ---

    img_tensor = img_tensor.cpu()
    image_after_mask = mask_img_tensor(
        img_tensor, f'{MASK_FOLDER}/{imagename}.png', 0)
    image_after_mask = diffusers.utils.numpy_to_pil(
        img_tensor_to_numpy(image_after_mask.unsqueeze(0)))[0]

    gt_mask = Image.open(
        f'{MASK_FOLDER}/{imagename}.png').convert("1")

    with open(f'{MASK_FOLDER}/{imagename}_info.json', 'r', encoding='utf-8') as f:
        obj = json.load(f)
        pos = obj["pos"]
    save_adjust_image(
        image_after_mask, gt_mask, pos, wm_path,
        image_after_mask_name, A=A)

    img_tensors: list[torch.Tensor] = list()
    mask2_tensors: list[torch.Tensor] = list()
    for iter in range(A):
        single_img_tensor = get_img_tensor(
            osp.join(
                wm_path, f'{image_after_mask_name}_pos_{iter+1}.png'), device,
            mask=f'{AFTER_MASK_FOLDER}/{imagename}_mask_{iter+1}.png',)
        single_mask_tensor = get_img_tensor(
            f'{AFTER_MASK_FOLDER}/{imagename}_mask2_{iter+1}.png',
            device=device, return_int=True)
        img_tensors.append(single_img_tensor)
        mask2_tensors.append(single_mask_tensor)

    with torch.no_grad():
        if BLOCK_BACKGROUND:
            input_for_process = torch.where(
                mask2_tensors[SPECIAL_POS_NUM] == 0,
                torch.zeros_like(img_tensors[SPECIAL_POS_NUM], device=device), img_tensors[SPECIAL_POS_NUM])
            input_for_process = torch.clamp(
                input_for_process, 0.0, 1.0)
        else:
            input_for_process = img_tensors[SPECIAL_POS_NUM]

        inputs = processor.process(input_for_process)
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        output_vector = outputs.last_hidden_state[0, 0, :L]
        del outputs
        output_score = torch.matmul(
            output_vector, fixed_direction_vectors.T,)

        info_revealed = torch.greater(output_score, 0.0).int()
        error_bits = (info_revealed != info_to_mask).sum()
        error_rate = error_bits/MESSAGE_BITS

    logger.info(
        f'SSIM {ssim_value}, PSNR, {psnr_value}, Error Rate: {error_rate}')

# %% [markdown]
    # ## Attack Watermarked Image with Individual Attacks

    # %%
    logger.info(f'===== Init Attackers =====')
    att_pipe = ReSDPipeline.from_pretrained(
        "./stable-diffusion-2-1-base/", torch_dtype=torch.float16)
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)

    CONTRAST_ATTACK_MASK_FOLDER = \
        "./image_mask_reduced/mask_reduced"

    attackers = {
        'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
        'diff_attacker_30': DiffWMAttacker(att_pipe, batch_size=5, noise_step=30, captions={}),
        'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
        'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.8': ContrastAttackerWithMask(CONTRAST_ATTACK_MASK_FOLDER, contrast=0.8),
        'contrast_1.2': ContrastAttackerWithMask(CONTRAST_ATTACK_MASK_FOLDER, contrast=1.2),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        # 'bm3d': BM3DAttacker(),
        'No': NoAttacker(),
    }

    # %%
    logger.info(f'===== Start Attacking... =====')

    post_img = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")
    for attacker_name, attacker in attackers.items():
        print(f'Attacking with {attacker_name}')
        os.makedirs(os.path.join(wm_path, attacker_name), exist_ok=True)
        att_img_path = os.path.join(
            wm_path, attacker_name, os.path.basename(post_img))

        current_attacker = attackers[attacker_name]
        if ("contrast" in attacker_name):
            assert isinstance(current_attacker, ContrastAttackerWithMask)
            mask_img = osp.join(
                CONTRAST_ATTACK_MASK_FOLDER, f"{imagename}.png")
            current_attacker.attack([post_img], [mask_img], [att_img_path])
        else:
            current_attacker.attack([post_img], [att_img_path])

    # %% [markdown]
    # ## Attack Watermarked Image with Combined Attacks

    # %%
    case_list = ['w/o rot',]

    logger.info(f'===== Init Attackers(\'all\') =====')
    att_pipe = ReSDPipeline.from_pretrained(
        "./stable-diffusion-2-1-base/", torch_dtype=torch.float16)
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)

    # %%
    for case in case_list:
        print(f'Case: {case}')
        if case == 'w/ rot':
            raise NotImplementedError
        elif case == 'w/o rot':

            CONTRAST_ATTACK_MASK_FOLDER = \
                "./image_mask_reduced/mask_reduced"
            # 
            attackers_combined = {
                'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
                'diff_attacker_30': DiffWMAttacker(att_pipe, batch_size=5, noise_step=30, captions={}),
                'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
                'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
                'jpeg_attacker_50': JPEGAttacker(quality=50),
                'brightness_0.5': BrightnessAttacker(brightness=0.5),
                'contrast_0.8': ContrastAttackerWithMask(CONTRAST_ATTACK_MASK_FOLDER, contrast=0.8),
                'contrast_1.2': ContrastAttackerWithMask(CONTRAST_ATTACK_MASK_FOLDER, contrast=1.2),
                'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
                'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
                # 'bm3d': BM3DAttacker(),
                'No': NoAttacker(),
            }
            multi_name = 'all_norot'

        os.makedirs(os.path.join(wm_path, multi_name), exist_ok=True)
        att_img_path = os.path.join(
            wm_path, multi_name, os.path.basename(post_img))
        for iter_attack, (attacker_name, attacker) in enumerate(attackers_combined.items()):
            print(f'Attacking with No[{iter_attack}]: {attacker_name}')
            current_attacker = attackers[attacker_name]
            if ("contrast" in attacker_name):
                assert isinstance(current_attacker, ContrastAttackerWithMask)
                mask_img = osp.join(
                    CONTRAST_ATTACK_MASK_FOLDER, f"{imagename}.png")
                current_attacker.attack([post_img], [mask_img], [att_img_path])
            else:
                current_attacker.attack([post_img], [att_img_path])

    # %% [markdown]
    # ## Detect Watermark

    attack_folders = ['diff_attacker_60', 'diff_attacker_30', 'cheng2020-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50',
                        'brightness_0.5', 'contrast_0.8', 'contrast_1.2', 'Gaussian_noise', 'Gaussian_blur',
                         'all_norot', 'No']
    logger.info(f'===== Testing the Watermarked Images {post_img} =====')

    #  det_prob
    error_rate_dict = {"Original": error_rate.item()}

    # %%
    logger.info(f'===== Testing the Attacked Watermarked Images =====')
    # 

    for attacker_name in attack_folders:
        attacked_dir = os.path.join(wm_path, attacker_name)
        if not os.path.exists(attacked_dir):
            logger.info(f'Attacked images under {attacker_name} not exist.')
            continue

        logger.info(f'=== Attacker Name: {attacker_name} ===')

        base_name = f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final"
        attacked_img_full_path = os.path.join(attacked_dir, base_name + ".png")

        if not os.path.exists(attacked_img_full_path):
            logger.warning(
                f"Image {attacked_img_full_path} not found. Skipping.")
            continue

        # 
        image_after_mask = get_img_tensor(
            attacked_img_full_path, device,
            mask=f'{MASK_FOLDER}/{imagename}.png', mask_value=0,
        )

        # PyTorch TensorPIL Image
        # : pipe.numpy_to_pil numpy
        pil_image_after_mask = transforms.ToPILImage()(image_after_mask)

        gt_mask = Image.open(f'{MASK_FOLDER}/{imagename}.png').convert("1")

        save_adjust_image(
            pil_image_after_mask, gt_mask, pos, attacked_dir, base_name, A=A,
        )

        img_tensors: list[torch.Tensor] = list()
        mask2_tensors: list[torch.Tensor] = list()
        for iter_pos in range(A):
            single_img_tensor = get_img_tensor(
                osp.join(attacked_dir,
                         f'{base_name}_pos_{iter_pos+1}.png'), device,
                mask=f'{AFTER_MASK_FOLDER}/{imagename}_mask_{iter_pos+1}.png',
            )
            single_mask_tensor = get_img_tensor(
                f'{AFTER_MASK_FOLDER}/{imagename}_mask2_{iter_pos+1}.png',
                device, return_int=True)
            img_tensors.append(single_img_tensor)
            mask2_tensors.append(single_mask_tensor)

        with torch.no_grad():
            if BLOCK_BACKGROUND:
                input_for_process = torch.where(
                    mask2_tensors[SPECIAL_POS_NUM] == 0,
                    torch.zeros_like(img_tensors[SPECIAL_POS_NUM], device=device), img_tensors[SPECIAL_POS_NUM])
                input_for_process = torch.clamp(input_for_process, 0.0, 1.0)
            else:
                input_for_process = img_tensors[SPECIAL_POS_NUM]

            inputs = processor.process(input_for_process)
            inputs = inputs.unsqueeze(0)
            outputs = model(inputs)

            output_vector = outputs.last_hidden_state[0, 0, :L]
            output_score = torch.matmul(
                output_vector, fixed_direction_vectors.T)
            info_revealed = torch.greater(output_score, 0.0).int()
            error_bits = (info_revealed != info_to_mask).sum()
            current_error_rate = error_bits / MESSAGE_BITS

            logger.info(f'Error Rate: {current_error_rate.item()}')
            error_rate_dict[attacker_name] = current_error_rate.item()

    # 
    #  .get(key, None) 
    row_data = [imagename] + \
        [error_rate_dict.get(key) for key in attack_folders_for_header]

    # 
    formatted_row = []
    for item in row_data:
        if isinstance(item, float):
            formatted_row.append(f"{item:.5f}")
        else:
            formatted_row.append(item)

    with open(CSV_RESULT_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(formatted_row)
    logger.info(
        f"Results for {imagename} have been saved to {CSV_RESULT_PATH}.")

    # ---  ---
    # 
    logger.info(f"Finished processing {imagename}. Cleaning up resources.")
    del att_pipe, attackers, attackers_combined


# 
logger.info("All images processed. Script finished.")
