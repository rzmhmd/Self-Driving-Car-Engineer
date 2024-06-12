from utils import get_data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION
    ii = 0
    fig, ax = plt.subplots(4, 5)
    for i, observation in enumerate(ground_truth):
        # print(observation)
        file_name = observation["filename"]
     #   print(file_name)
        im = Image.open(r"./data/images/" + file_name)
        ax[i % 4, i % 5].imshow(im)
        for j, box_info in enumerate(observation['boxes']):
         #       print(box_info)
           #         print(observation['classes'][j])
            if observation['classes'][j] == 1:
                box_color = 'red'
            else:
                box_color = 'green'
  #          ax[i % 4, i % 5].add_patch(Circle((box_info[0], box_info[1]),50))
    #        print(box_info[0], box_info[1])
            ax[i % 4, i % 5].add_patch(Rectangle((box_info[1], box_info[0]), box_info[3]-box_info[1], box_info[2]-box_info[0],
                                                 edgecolor=box_color,
                                                 facecolor='none',
                                                 fill=False,
                                                 lw=1))
        ax[i % 4, i % 5].axis('off')
    plt.show()

# box_info[3]-box_info[1],box_info[2]-box_info[0]


if __name__ == "__main__":
    ground_truth, _ = get_data()
#    print(ground_truth)
   # print(len(ground_truth[0]))
    # print(ground_truth[0]['boxes'][0])

    viz(ground_truth)
