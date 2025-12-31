import torch
import torchvision.transforms.functional as F2
from torchvision.transforms import InterpolationMode


class BitImageProcessor:
    def __init__(self):
        # 
        self.crop_size = {"height": 224, "width": 224}
        self.do_center_crop = True
        self.do_convert_rgb = True
        self.do_normalize = True
        self.do_rescale = True
        self.do_resize = True
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        # PIL  Image.BICUBIC  torchvision  InterpolationMode.BICUBIC
        self.resample = InterpolationMode.BICUBIC
        self.rescale_factor = 0.00392156862745098  # 1 / 255
        self.size = {"shortest_edge": 256}

    def __call__(self, image_tensor):
        return self.process(image_tensor)

    def process(self, image_tensor):
        
        #  (1,C,H,W) 
        if len(image_tensor.shape) == 3:
            unsqueeze_flag = False
        elif len(image_tensor.shape) == 4:
            unsqueeze_flag = True
            image_tensor = image_tensor.squeeze(0)
        else:
            raise RuntimeError

        """
        

        :
            image_tensor (torch.Tensor):  (C, H, W) [0, 255]

        :
            torch.Tensor:  (C, H, W) [0, 1] 
        """
        # 
        if not image_tensor.is_floating_point():
            image_tensor = image_tensor.float()

        #  RGB
        if self.do_convert_rgb and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        # Rescale ( [0, 1])
        if self.do_rescale:
            image_tensor = image_tensor * self.rescale_factor

        # Resize ()
        if self.do_resize:
            _, h, w = image_tensor.shape
            scale = self.size["shortest_edge"] / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_tensor = F2.resize(
                image_tensor, [new_h, new_w], interpolation=self.resample)

        # Center Crop ()
        if self.do_center_crop:
            crop_height, crop_width = self.crop_size["height"], self.crop_size["width"]
            image_tensor = F2.center_crop(
                image_tensor, [crop_height, crop_width])

        # Normalize ()
        if self.do_normalize:
            image_tensor = F2.normalize(
                image_tensor, mean=self.image_mean, std=self.image_std)

        if unsqueeze_flag:
            image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
