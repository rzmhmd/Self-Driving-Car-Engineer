import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageStat

from utils import check_results


def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    means = []
    stds = []
    for pic_path in image_list:
        im = Image.open(pic_path).convert('RGB')
        all_stats = ImageStat.Stat(im)
        means.append(np.array(all_stats.mean))
        stds.append(np.array(np.sqrt(all_stats.var)))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION

    red_color = []
    green_color = []
    blue_color = []

    for path in image_list:
        im = np.array(Image.open(path).convert('RGB'))

        for ii in range(im.shape[0]):
            for jj in range(im.shape[1]):
                red_color.append(im[ii, jj, 0])
                green_color.append(im[ii, jj, 1])
                blue_color.append(im[ii, jj, 2])
    plt.figure()
    sns.kdeplot(red_color, color='r')
    sns.kdeplot(green_color, color='g')
    sns.kdeplot(blue_color, color='b')
    plt.show()


if __name__ == "__main__":
    image_list = glob.glob('data/images/*')
#    print(image_list[4])
    mean, std = calculate_mean_std(image_list)
    channel_histogram(image_list[:5])
    check_results(mean, std)
