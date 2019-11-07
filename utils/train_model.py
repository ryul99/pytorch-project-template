import os
import math
import tqdm
import torch
import itertools
import traceback

from .test_model import testing_model
from .utils import get_commit_hash
from model.model import Net


def train(args, pt_dir, chkpt_path, train_loader, test_loader, writer, logger, hp, hp_str):
    model = Net(hp).cuda()

    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp.train.adam.initlr)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    git_hash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']
        git_hash = checkpoint['git_hash']
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")
    else:
        logger.info("Starting new training run.")

    try:
        for epoch in itertools.count(init_epoch+1):
            model.train()
            loader = tqdm.tqdm(train_loader, desc='Train data loader')
            for spec, target in loader:
                spec = spec.cuda()
                target = target.cuda()
                output = model(spec)
                loss = model.get_loss(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.train_logging(loss, step)
                    loader.set_description('Loss %.02f at step %d' % (loss, step))
            if epoch % hp.log.chkpt_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%05d.pt' % (args.name, git_hash, epoch))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'git_hash': git_hash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

            testing_model(model, test_loader, writer, step, hp)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()