import torch


def subsequent_mask(shape):
    """
    :param shape: The shape for the mask
    :return: The mask of shape to apply to input sequence
    """
    mask = torch.triu(torch.ones(shape), diagonal=1)
    return mask == 0
