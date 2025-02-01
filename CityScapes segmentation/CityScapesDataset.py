import os
from PIL import Image

class CityScapesDataset:
    def __init__(self, image_dir):
        self.image_dir = image_dir

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        """ mask_path = os.path.join(
            self.mask_dir,
            self.images[index].replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png"),
        ) """
        image = Image.open(image_path)
        # mask = Image.open(mask_path)
        # print(np.unique(mask))
        return image  # , mask