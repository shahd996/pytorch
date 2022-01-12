import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def IoU(boxes_preds, boxes_labels, box_format="midpoint"):
    # boxes_preds - N x 4, where N is # boxes
    if box_format == 'midpoint':
        # [x,y,w,h]
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2 # N x 1
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 2:3] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 3:4] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 2:3] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 3:4] + boxes_labels[..., 3:4] / 2
        
    if(box_format == 'corners'):
        # [x1,y1,x2,y2]
        box1_x1 = boxes_preds[..., 0:1] # N x 1
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
        
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    
    # .clamp(0) is for the case when they DO NOT intersect. clamp(0) means set to 0 if it's less than 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    
#     print("intersection coordinates: [",x1.item(),y1.item(),x2.item(),y2.item(),"]")
#     print("intersection:",intersection.item())
#     print("union:",union.item())
    
    return (intersection / (union + 1e-6))

def nms(bboxes, iou_threshold, threshold, box_format="corners"):
#     print("IoU threshold:",iou_threshold)
#     print("prob threshold:",threshold,'\n')
    
    # bboxes = [[class, Pc, x1, y1, x2, y2],[...],....]
    assert type(bboxes) == list
    
    # Bounding Boxes after Non-Max Supression to be returned
    bboxes_after_nms = []
    
    # 1) Discard all bounding boxes with Pc < prob_threshold (leave only >= prob_threshold)
    bboxes = [box for box in bboxes if box[1] >= threshold]
    
    # Sort bboxes for our conveinience
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # sort bboxes with highest probability at the beginning
#     print("Bounding Boxes after dropping <prob_threshold:")
    
    
#     for b in bboxes:
#         print(b)
#     print('\n')
    cnt = 0
    
    while bboxes:
        # Select the bounding box with the highest probability
        highest_prob_box = bboxes.pop(0)
#         print(f'Highest Prob Box {cnt}: {highest_prob_box}')
        
        # Filter - Leave only 1) bbox with difference class & 2) IOU < iou_threshold
        bboxes = [ 
            box for box in bboxes
            if 
            box[0] != highest_prob_box[0] # We don't want to remove different classes
            or # Keep only boxes with iou < threshold
            IoU(torch.tensor(highest_prob_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]
        
        bboxes_after_nms.append(highest_prob_box)
        cnt += 1
    
    return bboxes_after_nms 

# def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    
#     # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], [...], ...]
#     average_precisions = []
#     epsilon = 1e-6
    
#     for c in range(num_classes):
#         detections = []
#         ground_truths = []
        
#         for detection in pred_boxes:
#             if detection[1] == c:
#                 detections.append(detection)
        
#         for true_box in true_boxes:
#             if true_box[1] == c:
#                 ground_truths.append(true_box)
        
#         # ex)
#         # img 0 has 3 bboxes
#         # img 1 has 5 bboxes
#         # As a result: amount_bboxes ={0:3, 1:5}
#         amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
#         for key, val in amount_bboxes.items():
#             amount_bboxes[key] = torch.zeros(val)
            
#         # amount_boxes = {0:torch.tesnor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        
#         detections.sort(key=lambda x: x[2], reverse=True)
#         TP = torch.zeros((len(detections)))
#         FP = torch.zeros((len(detections)))
#         total_true_bboxes = len(ground_truths)
        
#         if total_true_bboxes == 0:
#             continue
        
#         for detection_idx, detection in enumerate(detections):
#             ground_truth_img = [
#                 bbox for bbox in ground_truths if bbox[0] == detection[0]
#             ]
            
#             num_gts = len(ground_truth_img) # number of target bounding boxes in this image
            
#             best_iou = 0
            
#             for idx, gt in enumerate(ground_truth_img):
#                 iou = IoU(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = idx
            
#             # Now we have a single bbox for particular class in a partiualr image
            
#             if best_iou > iou_threshold:
#                 if amount_bboxes[detection[0]][best_gt_idx] == 0: # This target bounding box has not yet been visited
#                     TP[detection_idx] = 1
#                     amount_bboxes[detection[0]][best_gt_idx] = 1 # Visited
#                 else: # If already visited
#                     FP[detection_idx] = 1
#             else:
#                 FP[detection_idx] = 1
                
#         # [1, 1, 0, 1, 0] -> [1,2,2,3,3]
#         TP_cumsum = torch.cumsum(TP, dim = 0) # cumsum -> prefix sum
#         FP_cumsum = torch.cumsum(FP, dim = 0)
#         recalls = TP_cumsum / (total_true_bboxes + epsilon)
#         precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        
#         precisions = torch.cat((torch.tensor([1]), precisions)) # we need to have this for numerical integration
#         recalls = torch.cat((torch.tensor([0]), recalls))
        
#         average_precisions.append(torch.trapz(precisions, recalls)) # trapz takes y, x
        
#     return sum(average_precisions) / len(average_precisions) 
def mAP(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)




def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cpu",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])