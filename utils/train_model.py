import math
import os

from utils.utils import get_logger, is_logging_process


def train_model(cfg, model, train_loader, writer):
    logger = get_logger(cfg, os.path.basename(__file__))
    model.net.train()
    for model_input, model_target in train_loader:
        model.optimize_parameters(model_input, model_target)
        loss = model.log.loss_v
        model.step += 1

        if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
            logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % cfg.log.summary_interval == 0:
            if writer is not None:
                writer.logging_with_step(loss, model.step, "train_loss")
            if is_logging_process():
                logger.info("Train Loss %.04f at step %d" % (loss, model.step))
