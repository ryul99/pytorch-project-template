import torch


def test_model(hp, model, test_loader, writer):
    model.net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for model_input, target in test_loader:
            model.feed_data(input=model_input, GT=target)
            output = model.run_network()
            loss_v = model.loss_f(output, model.GT)
            total_test_loss += loss_v.to("cpu").item()

        total_test_loss /= len(test_loader.dataset) / hp.test.batch_size

        if writer is not None and model.step:
            writer.test_logging(total_test_loss, model.step)
