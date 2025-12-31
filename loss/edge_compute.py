import torch
import torch.nn.functional as F
from main.utils import get_img_tensor, save_img  # 
import rich


def special_min_max_normalize(
    tensor: torch.Tensor, gamma: float = 1.0,
    min_max_value: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, float, float]:
    """
     [0, 1] 
    :param tensor: , (1, C, H, W)
    :return: ,
    """
    eps = 1e-2
    if isinstance(min_max_value, tuple):
        controlled_min_value, controlled_max_value = min_max_value
    else:
        min_value: float = tensor.min().item()
        max_value: float = tensor.max().item()
        difference_value: float = max_value - min_value
        if gamma < eps or gamma > 1-eps:
            clamp_tensor = tensor
            controlled_min_value, controlled_max_value = min_value, max_value
        else:
            controlled_min_value = min_value - eps
            controlled_max_value = max_value - gamma * difference_value

    clamp_tensor = torch.clamp_max(
        tensor, controlled_max_value
    )
    normalized_tensor: torch.Tensor = (
        clamp_tensor - controlled_min_value) / (controlled_max_value - controlled_min_value)

    if torch.min(normalized_tensor) < -0.1 or torch.max(normalized_tensor) > 1.1:
        raise RuntimeError("? ")
    return normalized_tensor, controlled_min_value, controlled_max_value


def sobel_edge_detection(image_tensor):
    """
     Sobel 
    :param image_tensor: , (1, C, H, W)
    :return: 
    """
    # 

    assert image_tensor.shape[0] == 1, "1"
    assert len(image_tensor.shape) == 4, " (1, C, H, W)"

    C = image_tensor.shape[1]

    #  Sobel 
    sobel_kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # 
    sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3).repeat(
        C, 1, 1, 1).to(image_tensor.device)
    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3).repeat(
        C, 1, 1, 1).to(image_tensor.device)

    #  Sobel 
    edge_image_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1, groups=C)
    edge_image_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1, groups=C)

    # 
    edge_image = torch.abs(edge_image_x) + torch.abs(edge_image_y)

    return edge_image
