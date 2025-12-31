from kornia.losses import SSIMLoss  #  Kornia  SSIM 
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
import logging
import lpips  #  LPIPS 
from torchmetrics import StructuralSimilarityIndexMeasure  #  SSIM

from loss.edge_compute import special_min_max_normalize, sobel_edge_detection

logger = logging.getLogger('terminal_logger')
# logger.setLevel(logging.DEBUG)


class LossProvider(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProvider, self).__init__()
        
        #  weights : [PSNR, LPIPS, Edge, Frequency]
        self.loss_weights = loss_weights 
        self.loss_psnr = self.psnr
        self.loss_lpips = lpips.LPIPS(net='alex').to(device)

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2) + 1e-10
        max_pixel = 1.0
        psnr_value = 10 * (-torch.log10(mse))
        return psnr_value

    def edge_loss(self, img1, img2, mask:torch.Tensor|None=None):
        img1_edge = sobel_edge_detection(img1)
        img2_edge = sobel_edge_detection(img2)
        if isinstance(mask,torch.Tensor):
            img1_edge_black = torch.zeros_like(img1_edge,device=img1.device)
            img1_edge = torch.where(mask==0,img1_edge_black,img1_edge)
            img2_edge = torch.where(mask==0,img1_edge_black,img2_edge)

        img1_edge=torch.where(img1_edge>0.2,0,5*img1_edge)
        img2_edge=torch.where(img2_edge>0.2,0,5*img2_edge)
        
        ans = torch.sum(torch.abs((img1_edge - img2_edge))) / torch.sum(img1_edge<0.2)
        return ans

    def forward(self, pred_img_tensor, gt_img_tensor,
                mask_img_tensor: torch.Tensor | None = None, 
                mask2_img_tensor: torch.Tensor | None = None, 
                do_log: bool = False):
        
        # --- Mask  ---
        MASK = False 
        white_area = pred_img_tensor

        # 1. PSNR Loss ( PSNR)
        lossP = - self.loss_psnr(white_area, gt_img_tensor)

        # 2. LPIPS Loss ( [-1, 1])
        pred_img_normalized = (white_area - 0.5) * 2
        gt_img_normalized = (gt_img_tensor - 0.5) * 2
        lossL = self.loss_lpips(pred_img_normalized, gt_img_normalized).reshape((1,))

        # 3. Edge Loss (Sobel)
        #  ()
        if len(self.loss_weights) > 2 and self.loss_weights[2] > 1e-5:
            loss_edge = self.edge_loss(gt_img_normalized, pred_img_normalized, mask2_img_tensor)
        else:
            loss_edge = torch.tensor([0.0], device=gt_img_tensor.device)

        # 
        #  index out of range get  list 
        w_p = self.loss_weights[0]
        w_l = self.loss_weights[1]
        w_e = self.loss_weights[2] if len(self.loss_weights) > 2 else 0.0

        loss = (
            lossP * w_p +
            lossL * w_l +
            loss_edge * w_e)

        if do_log:
            logger.info(
                f'PSNR {lossP.item():.4f}, LPIPS {lossL.item():.4f}, ' +
                f'Edge {loss_edge.item():.5f}' +
                f'Total {loss.item():.4f}'
            )
        return loss

# Freqmark  
class LossProviderAbilation(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProviderAbilation, self).__init__()

        self.loss_weights = loss_weights
        self.loss_psnr = self.psnr  # PSNR 
        self.loss_lpips = lpips.LPIPS(net='alex').to(device)  #  LPIPS 

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2) + 1e-8  # 
        max_pixel = 1.0
        psnr_value = 10 * torch.log10((max_pixel**2) / mse)
        return psnr_value

    def forward(self, pred_img_tensor, gt_img_tensor,
                mask_img_tensor: torch.Tensor | None = None, do_log: bool = False):

        MASK = False  # 
        if MASK:
            black_area = torch.where(
                mask_img_tensor == 0,
                pred_img_tensor, torch.zeros_like(pred_img_tensor)
            )
            white_area = torch.where(
                mask_img_tensor == 1,
                pred_img_tensor, torch.zeros_like(pred_img_tensor)
            )
        else:
            white_area = pred_img_tensor

        #  PSNR 
        lossP = - self.loss_psnr(white_area, gt_img_tensor)

        #  LPIPS  [-1, 1]  LPIPS
        pred_img_normalized = (white_area - 0.5) * 2  # [0, 1] -> [-1, 1]
        gt_img_normalized = (gt_img_tensor - 0.5) * 2      # [0, 1] -> [-1, 1]
        lossL = self.loss_lpips(
            pred_img_normalized, gt_img_normalized).reshape((1,))

        #  L2 
        if self.loss_weights[2] > 1e-2:
            lossL2 = self.edge_loss(
                pred_img_normalized, gt_img_normalized)
        else:
            lossL2 = torch.tensor([0.0,], device=gt_img_tensor.device)

        #  [PSNR, LPIPS, L2]
        loss = (
            lossP * self.loss_weights[0] +
            lossL * self.loss_weights[1] +
            lossL2 * self.loss_weights[2]
        )
        if do_log:
            logger.info(
                f'PSNR {lossP.item():.4f}, LPIPS {lossL.item():.4f},' +
                f' Edge {lossL2.item():.7f}, Total Loss {loss.item():.4f}'
            )
        return loss



class LossProviderWithFreqComponent(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProvider, self).__init__()
        
        #  weights : [PSNR, LPIPS, Edge, Frequency]
        self.loss_weights = loss_weights 
        self.loss_psnr = self.psnr
        self.loss_lpips = lpips.LPIPS(net='alex').to(device)

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2) + 1e-8
        max_pixel = 1.0
        psnr_value = 10 * torch.log10((max_pixel**2) / mse)
        return psnr_value

    def edge_loss(self, img1, img2):
        img1_edge = sobel_edge_detection(img1)
        img2_edge = sobel_edge_detection(img2)

        img1_edge, img1_control_min, img1_control_max = \
            special_min_max_normalize(img1_edge, 0.5)
        img2_edge, _, _ = special_min_max_normalize(
            img2_edge, 0.5, (img1_control_min, img1_control_max))

        ans = torch.mean(torch.abs((img1_edge - img2_edge)))
        return ans

    def frequency_loss(self, img1, img2):
        """
         (Spectral Loss)
         torch.fft.rfft2  2D 
        """
        # 1.  (Batch, Channel, H, W) -> (Batch, Channel, H, W/2 + 1) 
        # norm='ortho' 
        fft1 = torch.fft.rfft2(img1, norm='ortho')
        fft2 = torch.fft.rfft2(img2, norm='ortho')

        # 2.  (Magnitude): sqrt(real^2 + imag^2)
        mag1 = torch.abs(fft1)
        mag2 = torch.abs(fft2)

        # 3. Log  ()
        #  log 
        # 
        mag1_log = torch.log(mag1 + 1e-8)
        mag2_log = torch.log(mag2 + 1e-8)

        # 4.  L1 
        return torch.mean(torch.abs(mag1_log - mag2_log))

    def forward(self, pred_img_tensor, gt_img_tensor,
                mask_img_tensor: torch.Tensor | None = None, do_log: bool = False):
        
        # --- Mask  ---
        MASK = False 
        if MASK:
            black_area = torch.where(
                mask_img_tensor == 0,
                pred_img_tensor, torch.zeros_like(pred_img_tensor)
            )
            white_area = torch.where(
                mask_img_tensor == 1,
                pred_img_tensor, torch.zeros_like(pred_img_tensor)
            )
        else:
            white_area = pred_img_tensor

        # 1. PSNR Loss ( PSNR)
        lossP = - self.loss_psnr(white_area, gt_img_tensor)

        # 2. LPIPS Loss ( [-1, 1])
        pred_img_normalized = (white_area - 0.5) * 2
        gt_img_normalized = (gt_img_tensor - 0.5) * 2
        lossL = self.loss_lpips(pred_img_normalized, gt_img_normalized).reshape((1,))

        # 3. Edge Loss (Sobel)
        #  ()
        if len(self.loss_weights) > 2 and self.loss_weights[2] > 1e-5:
            loss_edge = self.edge_loss(gt_img_normalized, pred_img_normalized)
        else:
            loss_edge = torch.tensor([0.0], device=gt_img_tensor.device)

        # 4. Frequency Loss ()
        # 4
        if len(self.loss_weights) > 3 and self.loss_weights[3] > 1e-5:
            #  [0, 1] 
            loss_freq = self.frequency_loss(white_area, gt_img_tensor)
        else:
            loss_freq = torch.tensor([0.0], device=gt_img_tensor.device)

        # 
        #  index out of range get  list 
        w_p = self.loss_weights[0]
        w_l = self.loss_weights[1]
        w_e = self.loss_weights[2] if len(self.loss_weights) > 2 else 0.0
        w_f = self.loss_weights[3] if len(self.loss_weights) > 3 else 0.0

        loss = (
            lossP * w_p +
            lossL * w_l +
            loss_edge * w_e +
            loss_freq * w_f
        )

        if do_log:
            logger.info(
                f'PSNR {lossP.item():.4f}, LPIPS {lossL.item():.4f}, ' +
                f'Edge {loss_edge.item():.5f}, Freq {loss_freq.item():.5f}, ' +
                f'Total {loss.item():.4f}'
            )
        return loss

