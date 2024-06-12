import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # IMPLEMENT THIS FUNCTION
   # print(path)
    im = np.array(Image.open(path).convert('RGB'))
    mask = np.array(np.zeros((im.shape[0], im.shape[1])))

    for ii in range(im.shape[0]):
        for jj in range(im.shape[1]):
            if im[ii, jj, 0] > color_threshold[0] and im[ii, jj, 1] > color_threshold[1] and im[ii, jj, 2] > color_threshold[2]:
                mask[ii, jj] = 1

    img = im
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # IMPLEMENT THIS FUNCTION
    masked_im = img.copy()
#    masked_im = np.array(np.zeros((img.shape[0], img.shape[1], 3)))
    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            if mask[ii, jj] == 1:
                masked_im[ii, jj, 0] = img[ii, jj, 0]
                masked_im[ii, jj, 1] = img[ii, jj, 1]
                masked_im[ii, jj, 2] = img[ii, jj, 2]
            else:
                masked_im[ii, jj, 0] = 0
                masked_im[ii, jj, 1] = 0
                masked_im[ii, jj, 2] = 0

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[1].imshow(mask)
    axs[2].imshow(masked_im)
    plt.show()


if __name__ == '__main__':

    # print(os.path.exists(/data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png))

    path = '/data/images/segment-12273083120751993429_7285_000_7305_000_with_camera_labels_92.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)
