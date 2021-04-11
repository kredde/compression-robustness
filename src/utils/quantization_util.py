import os
import torch


def calibrate_model(model, criterion, data_loader, neval_batches):
    """
        Calibrates a given model with a given number of batches
    """
    model.eval()
    cnt = 0

    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            if cnt >= neval_batches:
                return


def get_model_size(model):
    """
      Returns the model size in MB
    """

    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size
