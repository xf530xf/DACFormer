import torch
import numpy as np
from DAC.transformer_seg import SETRModel
from PIL import Image
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def predict():
    model = build_model()
    model.load_state_dict(torch.load("./checkpoints/cell3.pkl", map_location="cpu"))
    print(model)

    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img, mask in val_loader:
            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).int()
            ax1=plt.subplot(1, 3, 1)
            ax1.set_title('input')
            print(img.shape)
            img = img.permute(0, 2, 3, 1)
            plt.imshow(img[0])
            ax2=plt.subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(pred[0].squeeze(0), cmap="gray")
            ax3=plt.subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(mask[0], cmap="gray")
            plt.show()


if __name__ == '__main__':

    def build_model():
        model = SETRModel(patch_size=(32, 32),
                          in_channels=3,
                          out_channels=1,
                          hidden_size=1024,
                          num_hidden_layers=6,
                          num_attention_heads=16,
                          decode_features=[512, 256, 128, 64])
        return model


    class CarDataset(Dataset):
        def __init__(self, img_url, mask_url):
            super(CarDataset, self).__init__()
            self.img_url = img_url
            self.mask_url = mask_url

        def __getitem__(self, idx):
            img = Image.open(self.img_url[idx]).convert("RGB")
            img = img.resize((512, 512))
            img_array = np.array(img, dtype=np.float32) / 255
            mask = Image.open(self.mask_url[idx])
            mask = mask.resize((512, 512))
            mask = np.array(mask, dtype=np.float32)
            img_array = img_array.transpose(2, 0, 1)

            return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

        def __len__(self):
            return len(self.img_url)


    # val_img_url = "/media/xf/新加卷/PyProject/datasets/isbi/test"
    # val_mask_url = ""

    # img_url = sorted(glob.glob("/media/xf/新加卷/PyProject/SETR-pytorch/data/segmentation_car/train/*"))
    # mask_url = sorted(glob.glob("/media/xf/新加卷/PyProject/SETR-pytorch/data/segmentation_car/train_masks/*"))
    img_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/cell2/test/image/*"))
    mask_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/cell2/test/label/*"))
    # img_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/isic/train/image/*"))
    # mask_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/isic/train/label/ISIC2018_Task1_Training_GroundTruth/*"))

    train_size = int(len(img_url) * 0.8)
    train_img_url = img_url[:train_size]
    train_mask_url = mask_url[:train_size]
    val_img_url = img_url[train_size:]
    val_mask_url = mask_url[train_size:]

    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    predict()
