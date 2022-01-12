import torch
import torch.nn as nn
from utils import IoU

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5 # For Confidence Loss - NO OBJECT CASE
        self.lambda_coord = 5 # For Localization Loss
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) # (m,S,S,30)

        # For each cell, we predict TWO BOUNDING BOXES. But ONLY ONE of them is responsible for the 
        # actual prediction. We pick this by choosing the bbox with higher IoU with the ground truth
        iou_b1 = IoU(predictions[..., 21:25], target[..., 21:25]) # only x,y,w,h
        iou_b2 = IoU(predictions[..., 26:30], target[..., 26:30]) # only x,y,w,h

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        ious_maxes, best_box = torch.max(ious, dim=0)

        # As we know, many of the S x S grid cells MIGHT NOT contain any object! So 
        # target -> (m, S, S, 25)
        # If we pick target[..., 20], it will become (m, S, S)
        # To match up dimension, we unsqueeze(3) so that it becomes (m, S, S, 1) again!!!
        exists_box = target[..., 20].unsqueeze(3) # Iobj_i from the paper for Loss Function, 0 or 1 

        # ======================== #
        #   LOCALIZATION LOSS      #
        # ======================== #
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30]) # best_box is 1 or 0
            +
            (1 - best_box) * predictions[...,21:25]
        )
        # box_predictions: (m, S, S, 4)

        box_targets = exists_box * target[..., 21:25]
        # box_targets: (m, S, S, 4)

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # 2:4 -> w,h

        # (N, S, S, 4) -> (N*S*S, 4)
        localization_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # ======================== #
        #   CONFIDENCE LOSS        #
        # ======================== # 

        # 1) If OBJECT DETECTED  
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21] 
        ) # pred_box: (m,S,S,1) -> for ONE responsible bounding box for each cell

        # (N*S*S)
        confidence_object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # 2) IF NO OBJECT DETECTED

        # (N,S,S,1) -> (N,S*S)
        confidence_no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )

        confidence_no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 25:26], start_dim = 1)
        )

        # ======================== #
        #   CLASSIFICATION LOSS    #
        # ======================== #

        # (N, S, S, 20) -> (N*S*S, 20)
        classification_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim = -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )


        loss = (
            self.lambda_coord * localization_loss # First two rows of loss in paper
            + confidence_object_loss
            + self.lambda_noobj * confidence_no_object_loss
            + classification_loss
        )

        return loss
            