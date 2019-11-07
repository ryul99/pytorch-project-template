import tqdm
import torch


def testing_model(model, test_loader, writer, step, hp):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for model_input, target in tqdm.tqdm(test_loader):
            target = target.cuda(); model_input = model_input.cuda()
            output = model.inference(model_input)
            total_test_loss += model.get_loss(output, target)

        total_test_loss /= (len(test_loader.dataset) / hp.test.batch_size)

        writer.test_logging(total_test_loss, step)
