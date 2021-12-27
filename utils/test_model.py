import os

import torch

from utils.utils import get_logger, is_logging_process


def test_model(cfg, model, test_loader, writer):
    logger = get_logger(cfg, os.path.basename(__file__))
    model.net.eval()
    total_test_loss = 0
    test_loop_len = 0
    with torch.no_grad():
        for model_input, model_target in test_loader:
            output = model.inference(model_input)
            loss_v = model.loss_f(output, model_target.to(cfg.device))
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                torch.distributed.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1

        total_test_loss /= test_loop_len

        if writer is not None:
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
        if is_logging_process():
            logger.info("Test Loss %.04f at step %d" % (total_test_loss, model.step))
