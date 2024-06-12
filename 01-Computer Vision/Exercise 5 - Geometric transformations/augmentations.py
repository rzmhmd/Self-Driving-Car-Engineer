import copy
import json
import random

import numpy as np
from PIL import Image

from utils import check_results, display_results


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])

    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMP8LEMENT THIS FUNCTION
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

    flipped_bboxes = np.array(bboxes)
    col_num = img.size[0]

    # for ii in range(flipped_img.size[1]):
    #     for jj in range(flipped_img.size[0]):
    #         flipped_img[ii, jj,0] = img[ii, col_num - jj,0]
    #         flipped_img[ii, jj,0] = img[ii, col_num - jj,0]
    #         flipped_img[ii, jj,0] = img[ii, col_num - jj,0]

    for ii in range(len(flipped_bboxes)):
        flipped_bboxes[ii][1] = col_num-bboxes[ii][3]
        flipped_bboxes[ii][3] = col_num - bboxes[ii][1]
    # print(flipped_bboxes)
    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    resized_image = img.resize(size)

    resized_boxes = boxes.copy()
    for ii, box in enumerate(resized_boxes):
        resized_boxes[ii][0] = boxes[ii][0]*(resized_image.size[1]/img.size[1])
        resized_boxes[ii][1] = boxes[ii][1]*(resized_image.size[0]/img.size[0])
        resized_boxes[ii][2] = boxes[ii][2]*(resized_image.size[1]/img.size[1])
        resized_boxes[ii][3] = boxes[ii][3]*(resized_image.size[0]/img.size[0])

    return resized_image, resized_boxes


def random_crop(img, boxes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    x1 = np.random.randint(0, img.size[0]-crop_size[0])
    y1 = np.random.randint(0, img.size[1]-crop_size[1])
    x2 = x1 + crop_size[0]
    y2 = y1 + crop_size[1]
    cropped_image = img.crop((x1, y1, x2, y2))
    cropped_boxes = []

    for ii, box in enumerate(boxes):
        calculate_iou_res = calculate_iou(box, [y1, x1, y2, x2])
        if calculate_iou_res[0] > 0 and (calculate_iou_res[1][0]-calculate_iou_res[1][2])*(calculate_iou_res[1][1]-calculate_iou_res[1][3]) > min_area:
            cropped_box = [calculate_iou_res[1][0] - y1, calculate_iou_res[1]
                           [1] - x1, calculate_iou_res[1][2] - y1, calculate_iou_res[1][3] - x1]
            cropped_boxes.append(cropped_box)
            # cropped_classes.append(bclass)

    return cropped_image, cropped_boxes


if __name__ == '__main__':
    # fix seed to check results
    np.random.seed()

    # open annotations
    with open('./data/ground_truth.json') as f:
        ground_truth = json.load(f)

    # filter annotations and open image
    file_num = np.random.randint(0, len(ground_truth))
    filename = 'segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png'
    bboxes = ground_truth[-1]['boxes']
    img = Image.open(f'./data/images/{filename}').convert('RGB')

    # check horizontal flip

    aug_img, aug_bboxes = hflip(img, bboxes)
    # display_results(img, bboxes, aug_img, aug_bboxes)
    # check_results(img, bboxes, aug_type='hflip')

    # check resize
    resized_image, resized_boxes = resize(img, bboxes, size=[640, 640])
    # display_results(img, bboxes, resized_image, resized_boxes)
    # check_results(img, bboxes, aug_type='resize', classes=resized_boxes)

    # # check random crop
    cropped_image, cropped_boxes = random_crop(
        img, bboxes, [512, 512], min_area=100)
    display_results(img, bboxes, cropped_image, cropped_boxes)
    check_results(cropped_image, cropped_boxes,
                  aug_type='random_crop')
