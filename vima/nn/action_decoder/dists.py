import torch


__all__ = ["Categorical", "MultiCategorical"]


class Categorical(torch.distributions.Categorical):
    def mode(self):
        return self.logits.argmax(dim=-1)


class MultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits, action_dims: list[int]):
        assert logits.dim() >= 2, logits.shape
        super().__init__(batch_shape=logits[:-1], validate_args=False)
        self._action_dims = tuple(action_dims)
        assert logits.size(-1) == sum(
            self._action_dims
        ), f"sum of action dims {self._action_dims} != {logits.size(-1)}"
        self._dists = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dims, dim=-1)
        ]

    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=-1) for dist in self._dists], dim=-1
        )
