# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
import torch
import numpy as np
from transformer_seg import SETRModel
from PIL import Image
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from  medpy import  metric
img_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/liver_kaggle/2d_images/*"))
mask_url = sorted(glob.glob("/media/xf/新加卷/PyProject/datasets/liver_kaggle/2d_masks/*"))


# print(img_url)
train_size = int(len(img_url) * 0.8)
train_img_url = img_url[:train_size]
train_mask_url = mask_url[:train_size]
val_img_url = img_url[train_size:]
val_mask_url = mask_url[train_size:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
epoches = 200
out_channels = 1


def build_model():
    model = DACFormerModel(patch_size=(32, 32),
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


def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t


# def computer_iou(input,target):
#     eps = 0.0001
#     if torch.is_tensor(input):
#         predict = torch.sigmoid(input).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#
#     return iou

def cmputer_iou(input,target):
    eps = 0.0001
    # input = (input > 0.5).float()
    # target = (target > 0.5).float()

    assert(len(input.shape)==len(target.shape))
    intersecion=np.multiply(input,target)
    union=np.asarray(input+target>0,np.float32)
    iou=intersecion.sum()/(union.sum()+1e-10)
    return iou

def Recall(predict, target): #Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall



def Precision(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fp = np.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision



def Specificity(predict, target): #Specificity，true negative rate一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


def Jac(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    intersection = np.count_nonzero(predict & target)
    union = np.count_nonzero(predict | target)

    jac = float(intersection) / float(union)

    return jac


# def predict():
#     model = build_model()
#     model.load_state_dict(torch.load("./checkpoints/kaggle_liver1.pkl", map_location="cpu"))
#     print(model)
#
#     import matplotlib.pyplot as plt
#     val_dataset = CarDataset(val_img_url, val_mask_url)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#     with torch.no_grad():
#         for img, mask in val_loader:
#             pred = torch.sigmoid(model(img))
#             pred = (pred > 0.5).int()
#             plt.subplot(1, 3, 1)
#             print(img.shape)
#             img = img.permute(0, 2, 3, 1)
#             plt.imshow(img[0])
#             plt.subplot(1, 3, 2)
#             plt.imshow(pred[0].squeeze(0), cmap="gray")
#             plt.subplot(1, 3, 3)
#             plt.imshow(mask[0], cmap="gray")
#             plt.show()


if __name__ == "__main__":

    model = build_model()
    model.to(device)

    train_dataset = CarDataset(train_img_url, train_mask_url)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    step = 0
    report_loss = 0.0
    for epoch in range(epoches):
        print("epoch is " + str(epoch))

        for img, mask in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)

            pred_img = model(img)  ## pred_img (batch, len, channel, W, H)

            if out_channels == 1:
                pred_img = pred_img.squeeze(1)  # 去掉通道维度
            print(pred_img.shape)
            print(mask.shape)
            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                dice = 0.0
                iou=0.0
                recall=0.0
                precision=0.0
                specificity=0.0
                jar=0.0
                n = 0
                model.eval()
                with torch.no_grad():
                    print("report_loss is " + str(report_loss))
                    report_loss = 0.0
                    for val_img, val_mask in tqdm(val_loader, total=len(val_loader)):
                        n += 1
                        val_img = val_img.to(device)
                        val_mask = val_mask.to(device)
                        pred_img = torch.sigmoid(model(val_img))
                        if out_channels == 1:
                            pred_img = pred_img.squeeze(1)
                        # pred_img=pred_img.cpu().numpy().flatten()
                        # val_mask=val_mask.cpu().numpy().flatten()c
                        # print(pred_img)
                        # print(val_mask)





                        cur_dice = compute_dice(pred_img, val_mask)
                        dice += cur_dice

                        # cur_iou=metric.binary.jc(pred_img, val_mask)
                        # # cur_iou =computer_iou(pred_img, val_mask)
                        # iou +=cur_iou
                        #
                        # cur_recall=metric.binary.recall(pred_img,val_mask)
                        # #cur_recall=Recall(pred_img, val_mask)
                        # recall +=cur_recall
                        #
                        # cur_pre=metric.binary.precision(pred_img,val_mask)
                        # #cur_pre=Precision(pred_img, val_mask)
                        # precision+=cur_pre
                        #
                        # cur_specificity=metric.binary.specificity(pred_img,val_mask)
                        # #cur_specificity=Specificity(pred_img, val_mask)
                        # specificity +=cur_specificity
                        # # cur_jar=Jac(pred_img, val_mask)
                        # # jar+=cur_jar
                    mdice = dice / n
                    # miou=iou/n
                    # mrecall=recall/n
                    # mprecision=precision/n
                    # mspecificity=specificity/n
                    # mjar=jar/n
                    print("mean dice is " + str(mdice))
                    # print("mean iou is "+str(miou))
                    # print("mean recall is " + str(mrecall))
                    # print("mean precision is " + str(mprecision))
                    # print("mean specificity is " + str(mspecificity))
                    # print("mean jar is " + str(mjar))
                    torch.save(model.state_dict(), "./checkpoints/kaggle_liver3.pkl")
                    model.train()
