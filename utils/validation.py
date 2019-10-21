import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def validate(model, testloader, writer, step):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for model_input, target in tqdm.tqdm(test_loader):
            target = target.cuda(); model_input = model_input.cuda()
            output = model.inference(model_input)
            total_test_loss += model.get_loss(output, target)

        total_test_loss /= (len(test_loader.dataset) / hp.test.batch_size)

        writer.validation_logging(total_test_loss, step)

    model.train()
