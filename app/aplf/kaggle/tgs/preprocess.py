from cytoolz.curried import keymap, filter, pipe, merge, map
from sklearn.metrics import jaccard_similarity_score
import pandas as pd
from skimage import io


def rl_enc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def mean_iou(y_preds, y_trues):
    score_sum = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        y_pred_image = io.imread(
            y_pred,
            as_gray=True
        ).reshape(-1)

        y_true_image = io.imread(
            y_true,
            as_gray=True
        ).reshape(-1)
        score_sum += jaccard_similarity_score(y_pred_image, y_true_image)
    return score_sum / len(y_preds)
