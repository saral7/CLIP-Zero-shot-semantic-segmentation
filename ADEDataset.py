import os
from PIL import Image


class ADEDataset:
    def __init__(self, dir, dirPred):
        self.image_dir = os.path.join(dir)
        self.mask_dir = os.path.join(dirPred)
        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[index],
        )
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # print(np.unique(mask))
        return image_path, image, mask
