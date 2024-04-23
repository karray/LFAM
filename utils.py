import numpy as np
from PIL import Image


def get_layer_idx(model, target_layer):
    for i, layer in enumerate(model.children()):
        if layer is target_layer:
            return i
    raise ValueError("Target layer not found in the model")


def normalize(masks):
    """
    Min-max normalization for each channel
    Args:
        masks (torch.Tensor): Masks BxCxWxH.

    """
    mins = masks.amin(dim=(-2, -1))
    maxs = masks.amax(dim=(-2, -1))
    # expand dims to [N activations, 1, 1]
    maxs, mins = maxs[:, None, None], mins[:, None, None]

    return (masks - mins) / (maxs - mins + 1e-8)


class SquareCropAndResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # Calculate the center crop size
        w, h = img.size
        new_size = min(w, h)

        # Center crop the image to a square
        left = (w - new_size) // 2
        top = (h - new_size) // 2
        right = (w + new_size) // 2
        bottom = (h + new_size) // 2
        img = img.crop((left, top, right, bottom))

        # Resize the image to the specified size
        img = img.resize((self.size, self.size), self.interpolation)

        return img


def evaluate(x_batch, s_batch, a_batch, quantus_metrics, device):
    a_batch = a_batch.cpu().numpy()
    x_batch = x_batch.cpu().numpy()
    s_batch = s_batch.cpu().numpy()
    s_batch = s_batch[:, np.newaxis, :, :]

    result = {}
    for metric in quantus_metrics:
        result[metric.name] = metric(disable_warnings=True)(
            model=None,
            x_batch=x_batch,
            y_batch=None,
            a_batch=a_batch,
            s_batch=s_batch,
            device=device,
        )[0]

    return result
