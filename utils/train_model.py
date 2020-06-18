import math


def train_model(hp, model, train_loader, writer, logger):
    model.net.train()
    for input_, target in train_loader:
        model.feed_data(input=input_, GT=target)
        model.optimize_parameters()
        loss = model.log.loss_v
        model.step += 1

        if logger is not None and (loss > 1e8 or math.isnan(loss)):
            logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % hp.log.summary_interval == 0:
            if writer is not None:
                writer.train_logging(loss, model.step)
            if logger is not None:
                logger.info("Train Loss %.04f at step %d" % (loss, model.step))
