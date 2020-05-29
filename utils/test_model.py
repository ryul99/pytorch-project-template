import tqdm
import torch


def test_model(hp, model, test_loader, writer):
    model.net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for model_input, target in tqdm.tqdm(test_loader):
            target = target.to(hp.model.device)
            model_input = model_input.to(hp.model.device)
            output = model.net(model_input)
            total_test_loss += model.get_loss(output, target)

        total_test_loss /= len(test_loader.dataset) / hp.test.batch_size

        writer.test_logging(total_test_loss, model.step)
