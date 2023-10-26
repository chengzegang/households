from typing import Type
from click import Tuple
from torch.utils.checkpoint import checkpoint
import torch
from torch import Tensor, nn
from functools import partial


def add_grad_checkpoint(
    module: nn.Module, module_types: Tuple[Type[nn.Module]], use_reentrant: bool = False
):
    def add_grad_checkpoint_(mod: nn.Module):
        if isinstance(mod, module_types):
            setattr(mod, "_org_forward", mod.forward)
            setattr(
                mod,
                "forward",
                partial(checkpoint, mod._org_forward, use_reentrant=use_reentrant),
            )
        return mod
    module.apply(add_grad_checkpoint_)
    return module
