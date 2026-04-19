import cv2
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_path']
        mask_path = self.df.iloc[index]['mask_path']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()
        mask[mask > 0] = 1.0

        return image, mask


def calculate_dice(logits, targets, eps=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    score = 2.0 * intersection / (preds.sum(dim=dims) + targets.sum(dim=dims) + eps)
    return score.mean().item()
