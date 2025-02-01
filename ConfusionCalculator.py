import os
from PIL import Image
from torchmetrics import JaccardIndex
from torchvision import transforms
import numpy as np
import torch
from torch import tensor
from torchmetrics.classification import MulticlassJaccardIndex
from ADEDataset import ADEDataset


def calc_mIoU(confusionMatrix):
    confusionMatrix = np.array(confusionMatrix)
    IoU = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(19):
        tp = confusionMatrix[i, i]
        fp = np.sum(confusionMatrix[i, :]) - confusionMatrix[i, i]
        fn = np.sum(confusionMatrix[:, i]) - confusionMatrix[i, i]
        # print(i, tp, fp, fn)
        if tp == 0:
            # print(0)
            continue
        IoU += tp / (tp + fp + fn)
        # print(tp / (tp + fp + fn), "%4f")
    return IoU / 19


def createConfusionMatrix(image, mask):
    mask = np.array(mask)
    image = np.array(image)
    valid_mask = mask != 255

    confusionMatrix = np.zeros((19, 19), dtype=int)
    np.add.at(confusionMatrix, (mask[valid_mask], image[valid_mask]), 1)
    return confusionMatrix


mIoUs_jacc = []
mIoUs = []

jaccard = JaccardIndex(
    task="multiclass", num_classes=151, average="macro", ignore_index=0
)


dataset = ADEDataset("annotations/validation", "predictions")

for i in range(len(dataset)):
    name, image, mask = dataset[i]
    print(i, name)

    transform = transforms.Compose([transforms.PILToTensor()])

    image = transform(image)[0]
    mask = transform(mask)[0]

    """ print("image", np.shape(image))
    print("mask", np.shape(mask))

    print(np.unique(image))
    print(np.unique(mask)) """
    """ np.savetxt("output.txt", image, fmt="%d")
      np.savetxt("mask.txt", mask, fmt="%d")
      confusionMatrix = createConfusionMatrix(image, mask)
      mIoU = calc_mIoU(confusionMatrix)
      print(mIoU) """
    """ valid_mask = mask != 0
    mask[mask == 255] = 19
    valid_mask = mask """

    jaccard.update(image, mask)
# mIoUs.append(mIoU)

mIoU = jaccard.compute()
print(mIoU)
# print(np.average(mIoUs))
