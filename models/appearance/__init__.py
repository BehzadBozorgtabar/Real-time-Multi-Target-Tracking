import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as Models
from functions import bbox as bbox_utils
from functions.log import logger
from models import net_utils
from models.appearance.resnet_model import Model
from distutils.version import LooseVersion


def appearance_model():
    model = Model()
    model = model.cuda()
    model.eval()
    return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_candidates(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    candidates = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return candidates


def appearance_features(appearance_model,image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    candidates = extract_candidates(image, tlbrs)
    candidates = np.asarray([im_preprocess(cv2.resize(c, (224,224))) for c in candidates], dtype=np.float32)
    gpu = net_utils.get_device(appearance_model)
    if LooseVersion(torch.__version__) > LooseVersion('0.3.1'):
        with torch.no_grad():
            im_var = Variable(torch.from_numpy(candidates))
            if gpu is not None:
                im_var = im_var.cuda(gpu)
            features = appearance_model(im_var).data

    else:
        im_var = Variable(torch.from_numpy(candidates), volatile=True)
        if gpu is not None:
            im_var = im_var.cuda(gpu)
        features = appearance_model(im_var).data

    return features
