from functions.nms.cpu_nms import cpu_nms
from functions.nms.gpu_nms import gpu_nms
import numpy as np

def nms(dets, thresh, force_cpu=False):

    if dets.shape[0] == 0:
        return []
    if force_cpu:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)


def nms_wrapper(pred_boxes, scores, nms_thresh):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    return keep
