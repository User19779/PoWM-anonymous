import torch
import torch.nn as nn
import torch.nn.functional as F

import rich

from main.block.InnBlock import Noise_INN_block
from main.block.InnBlock import attention_INN_block


class position_feature_INN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_num = 4

        self.INN_blocks: nn.ModuleList = nn.ModuleList(
            (attention_INN_block(clamp=1.0, input_1=4, input_2=4)
             for _ in range(self.block_num))
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], rev: bool = False):
        x1, x2 = x[0], x[1]
        x1_mid, x2_mid = x1, x2

        # 0INN_blocks
        #  rev 
        iterator_1 = reversed(range(self.block_num)
                              ) if rev else range(self.block_num)
        for i in iterator_1:
            x1_mid, x2_mid = self.INN_blocks[i].forward(
                (x1_mid, x2_mid), rev=rev)

        return x1_mid, x2_mid


class PositionSematicLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PositionSematicLoss, self).__init__()
        self.margin = margin  # Margin for contrastive loss

    def forward(
        self,
        position_fea: torch.Tensor, sematic_fea: torch.Tensor,
        original_fea: torch.Tensor, return_seperate_scores: bool = False
    ):
        """
        Args:
            position_fea: Tensor of shape [batch_size, channels, height, width]
            sematic_fea: Tensor of shape [batch_size, channels, height, width]
        Returns:
            loss: Scalar loss value
        """
        batch_size = position_fea.size(0)

        # Flatten the feature maps to compute pairwise distances
        pos_flat = position_fea.view(batch_size, -1)  # [B, C*H*W]
        sem_flat = sematic_fea.view(batch_size, -1)   # [B, C*H*W]
        original_flat = original_fea.view(batch_size, -1)

        POSITION_WEIGHT = 0.2

        if True:
            # Compute pairwise cosine similarity
            pos_sim = torch.nn.functional.cosine_similarity(
                pos_flat.unsqueeze(1), pos_flat.unsqueeze(0), dim=2)  # [B, B]
            sem_sim = torch.nn.functional.cosine_similarity(
                sem_flat.unsqueeze(1), sem_flat.unsqueeze(0), dim=2)  # [B, B]
            original_sim = torch.nn.functional.cosine_similarity(
                original_flat.unsqueeze(1), original_flat.unsqueeze(0), dim=2)  # [B, B]

            # Exclude diagonal elements (self-similarity)
            mask = ~torch.eye(batch_size, dtype=torch.bool,
                              device=pos_sim.device)
            pos_sim_masked = pos_sim[mask].view(
                batch_size, -1)  # Exclude self-similarities
            sem_sim_masked = sem_sim[mask].view(
                batch_size, -1)  # Exclude self-similarities
            original_sim_masked = original_sim[mask].view(
                batch_size, -1)  # Exclude self-similarities

            # Loss for position features: minimize similarity with other samples in the batch
            pos_loss = pos_sim_masked.mean()  # Negative mean similarity
            # Loss for semantic features: maximize similarity with other samples in the batch
            sem_loss = sem_sim_masked.mean()

            # TODO 
            original_loss = original_sim_masked.mean()

        else:
            pass

        # Total loss
        INN_loss = POSITION_WEIGHT * pos_loss + (-sem_loss)

        ADD_SIM_BETWEEN_ORIGINAL = True
        if ADD_SIM_BETWEEN_ORIGINAL:
            sim_1 = torch.nn.functional.cosine_similarity(
                pos_flat, original_flat, dim=1)
            sim_loss_1 = -sim_1.mean()

            original_flat_mean = torch.mean(
                original_flat, dim=(0,), keepdim=True)

            sim_2 = torch.nn.functional.cosine_similarity(
                sem_flat, original_flat_mean, dim=1)
            sim_loss_2 = -sim_2.mean()
            sim_loss = POSITION_WEIGHT * sim_loss_1 + sim_loss_2

        if return_seperate_scores:
            return (INN_loss, sim_loss, pos_loss, sem_loss, original_loss)
        else:
            return (INN_loss, sim_loss,)


# for testing only
if __name__ == "__main__":

    # 
    feature_shape = (8, 4, 100, 100)

    device = torch.device("cpu")
    input_1 = torch.rand(size=feature_shape, dtype=torch.float, device=device)
    input_2 = torch.zeros(size=feature_shape, dtype=torch.float, device=device)

    my_model = position_feature_INN()
    output_1, output_2 = my_model.forward((input_1, input_2), rev=True)
    rich.print(f"output shape 1:{output_1.shape}")
    rich.print(f"output shape 2:{output_2.shape}")
