import torch


def saliency(input, model):
    device = torch.device("cpu:0")
    model.to(device)
    input.to(device)

    for param in model.parameters():
        param.requires_grad = False

    input = input.reshape((1, 3, 32, 32))

    model.eval()

    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    return slc.numpy()
