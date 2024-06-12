import json
import os

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    # IMPLEMENT THIS FUNCTION
    for i, bi in enumerate(predictions['boxes']):
        for j, bj in enumerate(predictions['boxes']):
            if i != j and calculate_iou(bi, bj) > 0.5 and predictions['scores'][j] < predictions['scores'][i] and not (bi in filtered):
                filtered.append(list(bi, predictions['scores'][i]))
    return filtered


if __name__ == '__main__':
    # print(os.path.exists('data/predictions_nms.json'))
    # print(os.getcwd())
    os.chdir(r"C:\Users\mohamre1\Documents\python\udacity\Self Driving car engineer\Computer Vision\Exercise 12 - Non-Max Suppression")
    # print(os.getcwd())
    with open('data/predictions_nms.json') as f:
        predictions = json.load(f)

    filtered = nms(predictions)
    check_results(filtered)
