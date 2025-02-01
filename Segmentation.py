import torch
import open_clip
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ADEDataset
from torchvision.transforms import ToPILImage, transforms
import os
from PIL import Image
from open_clip import timm_model
import TimmChanges

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timm_model.TimmModel.forward = TimmChanges.myForward
timm_model.TimmModel.__init__ = TimmChanges.init

def myPreprocess(image):
    transform = transforms.Compose([transforms.PILToTensor()])

    # print(np.shape(image))
    image = transform(image).to(device)
    image = torch.unsqueeze(image, 0).float().to(device)

    mean, std = torch.mean(image), torch.std(image)
    # print(mean, std)
    image = (image - mean) / std
    image.to(device)
    ''' print(torch.mean(image), torch.std(image))

    print(np.shape(image)) '''
    return image

text_labels = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "window", "grass", "cabinet", "sidewalk", "person", "ground",
          "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug",
          "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
          "signboard", "chest", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway",
          "case", "pool", "pillow", "screen", "stairway", "river", "bridge", "bookcase", "blind", "coffee", "toilet", "flower", "book", "hill",
          "bench", "countertop", "stove", "palm tree", "kitchen island", "computer", "chair", "boat", "bar", "arcade", "hovel", "bus", "towel", "light",
          "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television", "airplane", "dirt", "apparel", "pole", "land",
          "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "belt", "canopy", "washer",
          "toy", "swimming", "stool", "barrel", "basket", "waterfall", "tent", "bag", "motorbike", "cradle",
          "oven", "ball", "food", "step", "tank", "brand name", "microwave", "pot", "animal",
          "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
          "vase", "traffic light", "tray", "garbage can", "fan", "pier", "crt screen", "plate",
          "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"]

text_labels = ["A photo of a " + className for className in text_labels]

#print(text_labels)

dataset = ADEDataset.ADEDataset("validation")

modelName = "convnext_base_w_320"
pretraining = "laion_aesthetic_s13b_b82k_augreg"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=modelName, pretrained=pretraining
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(modelName)



text = tokenizer(text_labels).to(device)

prediction = []
def doSegmentation (index):
    image = dataset[index]
    imageName = dataset.images[index]
    originalImage = image

    image = myPreprocess(image)
    print(imageName)
    print(np.shape(image))
    H = np.shape(image)[2]
    W = np.shape(image)[3]


    with torch.no_grad(): #, torch.amp.autocast(device):
        # print("model 2", model)
        # print("model encode", model.encode_image)
        image_features = model.encode_image(image).to(device)

        image_features = torch.nn.functional.interpolate(image_features, size=(H, W), mode="bilinear").to(device)
        text_features = model.encode_text(text).to(device)
        ''' print("Image features", np.shape(image_features))
        print("Text features", np.shape(text_features)) '''
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)


        ''' image_tensor = torch.from_numpy(np.array([image_features.cpu()]))
        print(np.shape(image_tensor))
        #print(image_tensor)



        image_features = torch.nn.functional.interpolate(
            torch.permute(image_tensor, (0, 3, 1, 2)),
            size=[H, W],
            mode="bilinear",
        ).to(device)
        print(np.shape(image_features))
        image_features = torch.permute(image_features, (0, 2, 3, 1))[0] '''

        ''' image_feat_cpu = image_features.cpu()
        print(np.shape(image_feat_cpu))

        text_feat_cpu = text_features.cpu()
        print(np.shape(text_feat_cpu)) '''

        image_features = torch.permute(image_features, (0, 2, 3, 1))[0]
        print("text", np.shape(text_features.T))
        print("image", np.shape(image_features))

        result = torch.matmul(image_features, text_features.T).to(device)
        result_softmax = torch.softmax(result, dim=-1).to(device)
        segments = torch.argmax(result_softmax, dim=-1).cpu() + 1


        print("result", np.shape(result))

        categories = Image.fromarray(np.array(segments, dtype=np.int32))
        categories.save("predictions/" + imageName.replace('jpg', 'png'))

        """ coloredImage = np.squeeze(coloredImage)
        coloredImage = coloredImage.astype(np.uint8)
        prediction = Image.fromarray(coloredImage) """
        # plt.imshow(prediction)
        #prediction.save("predictions_colors/" + imageName.replace('jpg', 'png'))
        #plt.figure(figsize = (80,7))
        """ plt.subplot(2, 1, 1).imshow(originalImage)
        plt.subplot(2, 1, 2).imshow(segments)
        plt.show()
        plt.savefig("predictions_colors/"+imageName.replace('jpg', 'png'))
        np.savetxt("predictions_text/"+imageName.replace('jpg', 'txt'), segments, fmt = "%d") """

for i in range(len(dataset)):
    print(i)
    doSegmentation(i)