import torch
import torch.nn.functional as F
from tqdm import tqdm


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
        interpolation="nearest",
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
        self.interpolation = interpolation
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

            heatmap = torch.zeros(shape, device=self.device)
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
                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=shape,
                    mode=self.interpolation,
                )

                masked = img * masks

                output_embs = self.model(masked)
                if self.unpack_output is not None:
                    output_embs = self.unpack_output(output_embs)

                score = self.score_function(target_embedding, output_embs)
                score = score[:, None, None, None]

                heatmap += (masks * score).sum(dim=0).squeeze(0)
                sum_of_weights += masks.sum(dim=0).squeeze(0)

            heatmaps.append(heatmap / sum_of_weights)

        heatmaps = torch.stack(heatmaps, dim=0)
        heatmaps = self._normalize(heatmaps)
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


# class RELAX:
#     def __init__(
#         self,
#         model,
#         score_function=torch.nn.CosineSimilarity(dim=1),
#         threshold=None,
#         unpack_output=None,
#         device="cuda",
#     ):
#         self.model = model
#         self.score_function = score_function
#         self.threshold = threshold
#         self.unpack_output = unpack_output
#         self.device = torch.device(device)

#     @torch.no_grad()
#     def __call__(self, x, n_masks, batch_size=16):
#         """
#         Args:
#             x (torch.Tensor): Input image BxCxWxH.
#             n_iterations (int, optional): Number of iterations for each image to compute importance and uncertainty maps. Default is 64.
#             n_masks (int, optional): Number of masks per iteration. Default is 64.
#             shift_cells (bool, optional): Whether to shift masks by a random number of pixels. Default is False.

#         Returns:
#             importance (numpy.ndarray): Importance map BxWxH.
#             uncertainty (numpy.ndarray): Uncertainty map BxWxH.
#         """

#         shape = x.shape[-2:]

#         importances = []
#         uncertainties = []

#         for img_i in range(x.shape[0]):
#             img = x[img_i].unsqueeze(0).to(self.device)

#             # fill tensor with 1e-10 to avoid division by zero
#             sum_of_weights = torch.zeros(shape, device=self.device) + 1e-10
#             importance = torch.zeros(shape, device=self.device)
#             # uncertainty = torch.zeros(shape, device=self.device)

#             target_embedding = self.model(img)
#             if self.unpack_output is not None:
#                 target_embedding = self.unpack_output(target_embedding)
#             target_embedding = target_embedding.squeeze(0)
#             assert (
#                 len(target_embedding.shape) == 1
#             ), "Target embedding must be 1-dimensional"

#             # for _ in tqdm(range(n_iterations), desc=f"Occlusion for image {img_i:03d}"):
#             #     for masks in self._generate_masks(n_masks, shape):
#             #         x_mask = img * masks
#             #         output_embs = self.model(x_mask)
#             #         if self.unpack_output is not None:
#             #             output_embs = self.unpack_output(output_embs)
#             #         if type(output_embs) == list:
#             #             output_embs = torch.cat(output_embs, dim=1)
#             #         assert (
#             #             len(output_embs.shape) == 2
#             #         ), "Output embeddings must be 2-dimensional (batch_size, embedding_size)"
#             #         scores = self.score_function(target_embedding, output_embs)
#             #         scores = torch.abs(scores)
#             #         print('scores.shape', scores.shape)
#             #         print('masks.shape', masks.shape)
#             #         print('sum_of_weights.shape', sum_of_weights.shape)
#             #         print('importance.shape', importance.shape)

#             #         for score, mask in zip(scores, masks.squeeze()):
#             #             sum_of_weights += mask

#             #             importance_prev = importance.clone()
#             #             importance += mask * (score - importance) / sum_of_weights
#             #             uncertainty += (
#             #                 (score - importance) * (score - importance_prev) * mask
#             #             )

#             # importances.append(importance.unsqueeze(0).cpu().numpy())
#             # uncertainties.append(
#             #     (uncertainty / (sum_of_weights - 1)).unsqueeze(0).cpu().numpy()
#             # )
#                 # (n_masks, 1, W, H)

#             for i in tqdm(range(0, n_masks, batch_size)):
#                 n_masks_batch = batch_size
#                 if i + batch_size > n_masks:
#                     n_masks_batch = n_masks - i

#                 masks = self._generate_masks(n_masks_batch, shape)
#                 x_masked = img * masks

#                 output_embs = self.model(x_masked)
#                 if self.unpack_output is not None:
#                     output_embs = self.unpack_output(output_embs)

#                 # (n_masks, )
#                 scores = self.score_function(target_embedding, output_embs)
#                 scores = (scores + 1) / 2


#                 masks = masks.squeeze(1)
#                 scores = scores.view(-1, 1, 1).expand_as(masks)

#                 sum_of_weights += masks.sum(dim=0)

#                 # importance_prev = importance.clone()

#                 importance_broadcasted = importance.unsqueeze(0).expand_as(masks)

#                 importance += (masks * (scores - importance_broadcasted) / sum_of_weights).sum(dim=0)

#                 # uncertainty += ((scores - importance) * (scores - importance_prev) * masks)


#             importances.append(importance.unsqueeze(0).cpu().numpy())
#             # uncertainties.append(
#             #     (uncertainty / (sum_of_weights - 1)).unsqueeze(0).cpu().numpy()
#             # )

#         importances = np.concatenate(importances)
#         # uncertainties = np.concatenate(uncertainties)

#         if self.threshold is not None:
#             importances[importances < self.threshold] = 0
#             # uncertainties[uncertainties < self.threshold] = 0

#         return importances

#     def _generate_masks(self, n_masks, shape, n_cells=7, p=0.5):
#         grid = torch.bernoulli(
#             torch.full((n_masks, 1, n_cells, n_cells), p, device=self.device)
#         )
#         return F.interpolate(grid, size=shape, mode="nearest")
