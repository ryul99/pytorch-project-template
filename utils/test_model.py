import torch


def test_model(hp, model, test_loader, writer, logger):
    model.net.eval()
    total_test_loss = 0
    test_loop_len = 0
    with torch.no_grad():
        for model_input, target in test_loader:
            model.feed_data(input=model_input, GT=target)
            output = model.run_network()
            loss_v = model.loss_f(output, model.GT)
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1

        total_test_loss /= test_loop_len

        if writer is not None:
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
        if logger is not None:
            logger.info("Test Loss %.04f at step %d" % (total_test_loss, model.step))
