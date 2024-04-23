import torch
import torch.nn.functional as F
from tqdm import tqdm
from captum.attr import LayerGradCam
from utils import normalize


class GradCAMHeatmap:
    def __init__(self, model, layer, imagenet_labels, interpolation, threshold=None):
        self.layer_gc = LayerGradCam(model, layer)
        self.imagenet_labels = imagenet_labels
        self.model = model
        self.interpolation = interpolation
        self.threshold = threshold

    def __call__(self, img_tensor):
        size = img_tensor.shape[-2:]
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = torch.argmax(pred, dim=1).item()
            pred_label = self.imagenet_labels[pred]
            pred_label = pred_label.split(",")[0]

        heatmap = self.layer_gc.attribute(
            img_tensor, int(pred), relu_attributions=True
        ).detach()
        heatmap = normalize(heatmap)
        if self.threshold is not None:
            heatmap[heatmap < self.threshold] = 0
        if self.interpolation is not None:
            heatmap = F.interpolate(heatmap, size=size, mode=self.interpolation)
        return heatmap.squeeze(0), pred_label


class LaFAM:
    def __init__(self, model, interpolation, threshold=None):
        self.model = model
        self.interpolation = interpolation
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, img_tensor, silent=None):
        size = img_tensor.shape[-2:]
        heatmap = self.model(img_tensor)
        heatmap = heatmap.mean(dim=1, keepdim=True)
        heatmap = normalize(heatmap)
        if self.threshold is not None:
            heatmap[heatmap < self.threshold] = 0
        if self.interpolation is not None:
            heatmap = F.interpolate(heatmap, size=size, mode=self.interpolation)
        return heatmap.squeeze(0)


class RELAX:
    """
    RELAX implementation.
    """

    def __init__(
        self,
        model,
        n_masks,
        n_cells,
        p=0.5,
        score_function=torch.nn.CosineSimilarity(dim=1),
        threshold=None,
        unpack_output=None,
        device="cuda",
        occlusion_batch_size=16,
        mask_interpolation="nearest",
        heatmap_interpolation="nearest",
    ):
        """
        Args:
            model (torch.nn.Module): Encoder model.
            target_layer (torch.nn.Module): Target layer for computing the saliency maps.
            score_function (torch.nn.Module, optional): Function for measuring similarity between embeddings of the original and occluded images.
            threshold (float, optional): Threshold for the saliency maps.
            unpack_output (callable, optional): Function for unpacking the output of the model.
            device (str, optional): Device where the computation will be performed.
            occlusion_batch_size (int, optional): Batch size for occlusion.
            interpolation (str, optional): Interpolation mode for masks.
        """
        self.device = torch.device(device)
        self.threshold = threshold
        self.occlusion_batch_size = occlusion_batch_size
        self.mask_interpolation = mask_interpolation
        self.heatmap_interpolation = heatmap_interpolation
        self.model = model
        self.model.to(self.device)
        self.unpack_output = unpack_output
        self.score_function = score_function
        self.n_masks = n_masks
        self.n_cells = n_cells
        self.p = p

    @torch.no_grad()
    def __call__(self, x, silent=False):
        """
        Args:
            x (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Saliency maps for each image in the batch.
        """

        heatmaps = []
        shape = x.shape[-2:]

        for img_idx in range(x.shape[0]):
            img = x[img_idx].to(self.device).unsqueeze(0)

            target_embedding = self.model(img)

            if self.unpack_output is not None:
                target_embedding = self.unpack_output(target_embedding)

            heatmap = torch.zeros((self.n_cells, self.n_cells), device=self.device)
            sum_of_weights = torch.zeros_like(heatmap) + 1e-10

            for i in tqdm(
                range(0, self.n_masks, self.occlusion_batch_size),
                desc=f"Occlusion for image {img_idx + img_idx:03d}",
                disable=silent,
            ):
                n_masks_batch = self.occlusion_batch_size
                if i + self.occlusion_batch_size > self.n_masks:
                    n_masks_batch = self.n_masks - i

                masks = self._generate_masks(n_masks_batch)
                masks_interpolated = F.interpolate(
                    masks.unsqueeze(1),
                    size=shape,
                    mode=self.mask_interpolation,
                )

                masked = img * masks_interpolated

                output_embs = self.model(masked)
                if self.unpack_output is not None:
                    output_embs = self.unpack_output(output_embs)

                score = self.score_function(target_embedding, output_embs)
                score = score[:, None, None]

                heatmap += (masks * score).sum(dim=0).squeeze(0)
                sum_of_weights += masks.sum(dim=0).squeeze(0)

            heatmaps.append(heatmap / sum_of_weights)

        heatmaps = torch.stack(heatmaps, dim=0)
        heatmaps = self._normalize(heatmaps)
        if self.heatmap_interpolation is not None:
            heatmaps = F.interpolate(
                heatmaps.unsqueeze(1),
                size=shape,
                mode=self.heatmap_interpolation,
            ).squeeze(1)
        if self.threshold is not None:
            heatmaps[heatmaps < self.threshold] = 0

        return heatmaps

    def _generate_masks(self, n_masks):
        return torch.bernoulli(
            torch.full(
                (n_masks, self.n_cells, self.n_cells),
                self.p,
                device=self.device,
            )
        )

    def _normalize(self, masks):
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
