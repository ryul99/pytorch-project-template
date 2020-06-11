import math
import tqdm


def train_model(hp, model, train_loader, writer, logger):
    model.net.train()
    loader = tqdm.tqdm(train_loader, desc="Train data loader")
    for input_, target in loader:
        model.feed_data(input=input_, GT=target)
        model.optimize_parameters()
        loss = model.log.loss_v
        model.step += 1

        if loss > 1e8 or math.isnan(loss):
            logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % hp.log.summary_interval == 0:
            writer.train_logging(loss, model.step)
            loader.set_description("Loss %.02f at step %d" % (loss, model.step))
