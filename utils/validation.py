import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def validate(model, testloader, writer, step):
    model.eval()

    with torch.no_grad():
        gt = list()
        scores = list()
        for target, spec1, spec2 in tqdm.tqdm(testloader):
            target = target.cuda(); spec1 = spec1.cuda(); spec2 = spec2.cuda()
            output1 = model.inference(spec1)
            output2 = model.inference(spec2)
            similarity = F.cosine_similarity(output1, output2)
            gt.append(target)
            scores.append(similarity)

        gt = torch.cat(gt, dim=0).tolist()
        scores = torch.cat(scores, dim=0).tolist()

        fpr, tpr, thresholds = roc_curve(gt, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thr = interp1d(fpr, thresholds)(eer)

        writer.log_validation(fpr, tpr, thresholds, eer, thr, step)

    model.train()
