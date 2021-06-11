import torch


def get_modules(model: torch.nn.Module):
    """
        Returns list of modules in a model
    """
    children = list(model.children())
    flat_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flat_children.extend(get_modules(child))
            except TypeError:
                flat_children.append(get_modules(child))
    return flat_children


def get_prunable_modules(model: torch.nn.Module):
    """
        Returns a list of prunable modules, i.e. all linear and convolutional layers
    """
    modules = get_modules(model)

    return [m for m in modules if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
