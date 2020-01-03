import numpy as np
from skimage.draw import rectangle
from torch.utils.data import Dataset


def rotate(image, rot_angle):
    assert rot_angle in [0, 90, 180, 270]

    if rot_angle == 90:
        image = image.transpose(1, 0, 2)[::-1]
    elif rot_angle == 180:
        image = image[::-1, ::-1]
    elif rot_angle == 270:
        image = image.transpose(1, 0, 2)[:, ::-1]

    return image


def draw_object(color=(128, 128, 128), rot_angle=0, flip=False, pos=(8, 8),
                image_size=(32, 32), size=(16, 16), chop_size=(12, 4)):
    img = np.zeros(image_size + (3,), dtype=np.uint8)
    rr, cc = rectangle(start=pos, extent=size, shape=image_size)
    img[rr, cc] = color
    rr, cc = rectangle(start=pos, extent=chop_size, shape=image_size)
    img[rr, cc] = 0

    if flip:
        img = img[::-1]

    return rotate(img, rot_angle)


class ChunkRectangleDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""

    def __init__(self, image_size=(32, 32), object_size=(16, 16), chop_size=(12, 4),
                 random_color=True, random_rot=True, random_flip=True, random_pos=True, transform=None):

        self.image_size = image_size
        self.object_size = object_size
        self.chop_size = chop_size

        self.random_color = random_color
        self.random_rot = random_rot
        self.random_flip = random_flip
        self.random_pos = random_pos

        self.transform = transform

    def __len__(self):
        return 2048

    def generate_object(self):
        if self.random_color:
            color = np.random.randint(256, size=3)
        else:
            color = (128, 128, 128)

        if self.random_flip:
            flip = np.random.randint(2)
        else:
            flip = False

        if self.random_rot:
            rot_angle = np.random.randint(4) * 90
        else:
            rot_angle = 0

        if self.random_pos:
            pos0 = np.random.randint(self.image_size[0] - self.object_size[0] + 1)
            pos1 = np.random.randint(self.image_size[1] - self.object_size[1] + 1)
            pos = (pos0, pos1)
        else:
            pos0 = (self.image_size[0] - self.object_size[0]) // 2
            pos1 = (self.image_size[1] - self.object_size[1]) // 2
            pos = (pos0, pos1)

        return draw_object(color, rot_angle, flip, pos, self.image_size, self.object_size, self.chop_size)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.generate_object()), 0
        else:
            return self.generate_object(), 0


if __name__ == "__main__":
    import imageio

    imageio.imsave('1.png', draw_object())

    print(np.random.randint(2, size=100))
