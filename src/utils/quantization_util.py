import os
import torch

UINT_BOUNDS = {8: [0, 255], 7: [0, 127], 6: [0, 63],
               5: [0, 31], 4: [0, 15], 3: [0, 7], 2: [0, 3]}
INT_BOUNDS = {8: [-128, 127], 7: [-64, 63], 6: [-32, 31],
              5: [-16, 15], 4: [-8, 7], 3: [-4, 3], 2: [-2, 1]}


def calibrate_model(model, criterion, data_loader, neval_batches):
    """
        Calibrates a given model with a given number of batches
    """
    model.eval()
    cpu = torch.device("cpu")
    
    cnt = 0

    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(cpu)
            target = target.to(cpu)
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
