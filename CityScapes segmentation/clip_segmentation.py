import open_clip
from open_clip import timm_model

import torch
import torch.nn as nn
import numpy as np
import Labels
import logging
from collections import OrderedDict

import TimmChanges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


timm_model.TimmModel.forward = TimmChanges.myForward
timm_model.TimmModel.__init__ = TimmChanges.init

import open_clip
import os
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, transforms
import CityScapesDataset

root = ""

dataset = CityScapesDataset(root)

modelName = "convnext_base_w_320"
pretraining = "laion_aesthetic_s13b_b82k_augreg"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=modelName, pretrained=pretraining
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(modelName)

index = 38
image = dataset[index]
imageName = dataset.images[index]
originalImage = image

plt.imshow(originalImage)

usedLabels = []
[usedLabels.append(label) for label in Labels.labels if label[2] != 255]
for i in range(len(usedLabels)):
    print("LABEL", Labels.trainId2label[i])
text_labels = ["A photo of a " + label[0] for label in usedLabels]
colors = [Labels.name2label[labelName[13:]].color for labelName in text_labels]
cMap = ListedColormap(colors)


def myPreprocess(image):
    transform = transforms.Compose([transforms.PILToTensor()])

    # print(np.shape(image))
    image = transform(image).to(device)
    image = torch.unsqueeze(image, 0).float().to(device)

    mean, std = torch.mean(image), torch.std(image)
    # print(mean, std)
    image = (image - mean) / std
    image.to(device)
    """ print(torch.mean(image), torch.std(image))

    print(np.shape(image)) """
    return image


text = tokenizer(text_labels).to(device)

prediction = []


def doSegmentation(index):
    image = dataset[index]
    imageName = dataset.images[index]
    originalImage = image

    image = myPreprocess(image)

    with torch.no_grad():  # , torch.amp.autocast(device):
        # print("model 2", model)
        # print("model encode", model.encode_image)
        image_features = model.encode_image(image).to(device)

        image_features = torch.nn.functional.interpolate(
            image_features, size=(1024, 2048), mode="bilinear"
        ).to(device)
        text_features = model.encode_text(text).to(device)
        """ print("Image features", np.shape(image_features))
        print("Text features", np.shape(text_features)) """
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        H = 1024
        W = 2048

        """ image_tensor = torch.from_numpy(np.array([image_features.cpu()]))
        print(np.shape(image_tensor))
        #print(image_tensor)



        image_features = torch.nn.functional.interpolate(
            torch.permute(image_tensor, (0, 3, 1, 2)),
            size=[H, W],
            mode="bilinear",
        ).to(device)
        print(np.shape(image_features))
        image_features = torch.permute(image_features, (0, 2, 3, 1))[0] """

        """ image_feat_cpu = image_features.cpu()
        print(np.shape(image_feat_cpu))

        text_feat_cpu = text_features.cpu()
        print(np.shape(text_feat_cpu)) """

        image_features = torch.permute(image_features, (0, 2, 3, 1))[0]
        print("text", np.shape(text_features.T))
        print("image", np.shape(image_features))

        result = torch.matmul(image_features, text_features.T).to(device)
        result_softmax = torch.softmax(result, dim=-1).to(device)
        segments = torch.argmax(result_softmax, dim=-1).cpu()

        print("result", np.shape(result))
        """ categories = Image.fromarray(np.array(segments, dtype=np.int32))
        categories.save(root+"/predictions_frankfurt/" + imageName) """

        coloredImage = np.array(
            [
                [Labels.trainId2label[int(pixelId)].color for pixelId in row]
                for row in segments
            ],
            dtype=np.int32,
        )

        coloredImage = np.squeeze(coloredImage)
        coloredImage = coloredImage.astype(np.uint8)
        prediction = Image.fromarray(coloredImage)
        # plt.imshow(prediction)
        prediction.save(root + "/predictions_colors/" + imageName)

        plt.figure(figsize=(80, 7))
        plt.subplot(2, 1, 1).imshow(originalImage)
        plt.subplot(2, 1, 2).imshow(prediction)


for i in range(0, 10):
    doSegmentation(i)
