import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    
def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform):
    train_dataset = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    
    val_dataset = CarvanaDataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform
    )
    
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    
    return train_loader, val_loader

def check_accuracy(loader, model, device='cpu'): # for binary - sigmoid
    num_correct = 0
    num_pixels = 0
    dice_score = 0 # metric for carvana
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score:{dice_score/len(loader)}")
    
    model.train()
    
    
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            print(preds)
            preds = (preds > 0.5).float()
            print(preds.shape)
            print(preds.sum())
            print(preds)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        print("image saved")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()