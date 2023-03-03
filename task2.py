import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes
from heapq import *
from scipy import interpolate


def calculate_iou(pt_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    intersection = 0
    if pt_box[0] < gt_box[2] and pt_box[2] >gt_box[0]:
        if pt_box[1] < gt_box[3] and pt_box[3]> gt_box[1]:
            
            x_min = max( gt_box[0], pt_box[0])
            x_max = min( gt_box[2], pt_box[2])
            y_min = max( gt_box[1], pt_box[1])
            y_max = min( gt_box[3], pt_box[3])
            
            intersection += (x_max - x_min) * ( y_max-y_min)

    union = (pt_box[2]- pt_box[0])*(pt_box[3]-pt_box[1]) + (gt_box[2]- gt_box[0])*(gt_box[3]-gt_box[1]) - intersection

    iou = 0 
    iou += intersection /union
    #END OF YOUR CODE

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # YOUR CODE HERE
    if (num_tp+num_fp) == 0: 
        return 1
    return num_tp/(num_tp+num_fp)
    #END OF YOUR CODE



def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    # YOUR CODE HERE
    if (num_tp + num_fn) == 0: 
        return  0
    return num_tp/(num_tp + num_fn)
    #END OF YOUR CODE



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    

    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.
        Remember: Matching of bounding boxes should be done with decreasing IoU order!
    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE
    matches = [] #This is going to be a heap
    # Find all possible matches with a IoU >= iou threshold
    for i in range(len(prediction_boxes)): 
        for j in range(len(gt_boxes)): 
            iuo = calculate_iou(prediction_boxes[i], gt_boxes[j])
            if iuo >= iou_threshold:
                heappush(matches, (iuo, i,j))  #i is the index of the prediction box and j is the index of the gt box
    #we are adding the negative of iuo since in the heapq is a min heap

    # Sort all matches on IoU in descending order
    prediction_match = []
    ground_match = []
    i_used = []
    j_used = []
    # Find all matches with the highest IoU threshold
    while len(matches): #This is true as long as our heap isn t empty
        match = list(matches.pop())
        if match[1] not in i_used: 
            if match[2] not in j_used: 
                prediction_match.append(prediction_boxes[match[1]])
                ground_match.append(gt_boxes[match[2]])
                j_used.append(match[2])
                i_used.append(match[1])

                                        

    return np.array(prediction_match), np.array(ground_match)
    #END OF YOUR CODE



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!
    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # YOUR CODE HERE
    pt_match, gt_match = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    TP = len(pt_match)
    FP = len(prediction_boxes) - TP
    FN = len(gt_boxes) - TP
    return {"true_pos": TP, "false_pos": FP, "false_neg": FN}
    #END OF YOUR CODE



def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # YOUR CODE HERE
    TP = 0
    FP = 0 
    FN = 0 
    for pred,ground in zip(all_prediction_boxes, all_gt_boxes): 
            
        results = calculate_individual_image_result(pred, ground, iou_threshold)
        TP += results["true_pos"]
        FP += results["false_pos"]
        FN += results["false_neg"]
    
    recall = calculate_recall(TP, FP, FN)
    precision = calculate_precision(TP, FP, FN)
    return precision, recall
    #END OF YOUR CODE



def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]
            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []

    for conf in confidence_thresholds:
        confident_predictions = []
        for pred,score in zip(all_prediction_boxes, confidence_scores):
            pred_image = []
            for box, s in zip(pred, score): 
                if s >= conf: 
                    pred_image.append(box)
            confident_predictions.append(pred_image)

        precision, recall= calculate_precision_recall_all_images(confident_predictions, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

        # END OF YOUR CODE

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'
    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.
    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # YOUR CODE HERE
    # Calculate the mean average precision given these recall levels.
    av = 0
    r_list = np.linspace(0,1,11)
    for r in r_list:
        p = precisions[recalls>=r]
        if len(p): 
            av += max(p)
    return av/11
    #END OF YOUR CODE



def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5
    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))

    plot_precision_recall_curve(precisions, recalls)

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
    
if __name__ == "__main__":
    main()