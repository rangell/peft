from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)


class UnsafeSteeringVectorLayer(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype, shared_reps: bool = False):
        super(UnsafeSteeringVectorLayer, self).__init__()
        self.dim = dim
        self.dtype = dtype
        self.shared_reps = shared_reps
        self.rep_steer_directions = nn.Parameter(torch.empty(dim, dtype=dtype).uniform_(-0.1, 0.1), requires_grad=True)

    def forward(self, x):
        _normed_unsafe_direction = F.normalize(self.rep_steer_directions, p=2, dim=0)
        x = x - torch.einsum("bl,d->bld", F.gelu(torch.matmul(x, _normed_unsafe_direction)), _normed_unsafe_direction)
        return x

    def __repr__(self) -> str:
        return f"dim={self.dim}, dtype={self.dtype}, shared_reps={self.shared_reps}"


class TiedUnsafeSteeringVectorLayer(nn.Module, BaseTunerLayer):
    def __init__(self, tied_module: nn.Module):
        super(TiedUnsafeSteeringVectorLayer, self).__init__()
        self.tied_module = tied_module

    def forward(self, x):
        _normed_unsafe_direction = F.normalize(self.tied_module.rep_steer_directions, p=2, dim=0)
        x = x - torch.einsum("bl,d->bld", F.relu(torch.matmul(x, _normed_unsafe_direction)), _normed_unsafe_direction)
        return x

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rep_steer." + rep


class MergedTransformerRepSteerLayer(nn.Module):
    def __init__(self, transformer_layer, steering_layer):
        super(MergedTransformerRepSteerLayer, self).__init__()
        self.transformer_layer = transformer_layer
        self.steering_layer = steering_layer

    def forward(self, *args, **kwargs):
        outputs = self.transformer_layer(*args, **kwargs)
        outputs = list(outputs)
        outputs[0] = self.steering_layer(outputs[0])
        return tuple(outputs)

    def __repr__(self) -> str:
        return f"transformer={self.transformer_layer.__repr__}, steering_layer={self.steering_layer.__repr__}"


class RepSteerModel(BaseTuner):
    prefix: str = "rep_steer_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        # TODO: fix later when more options are added
        return peft_config

    def _create_and_replace(
        self,
        rep_steer_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        assert rep_steer_config.target_modules is not None, "Target modules should not be `None`"
        assert rep_steer_config.shared_reps, "Shared should be `True` - unshared not implemented yet"
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if not hasattr(self, "steering_layers"):
            self.steering_layers = []
            self.steering_layer = UnsafeSteeringVectorLayer(
                dim=self.model.config.hidden_size,
                dtype=self.model.config.torch_dtype,
                shared_reps=rep_steer_config.shared_reps)
        
        # NOTE: to future user, we will used "Tied..." version of the steering layer even when weights aren't shared
        self.steering_layers.append(TiedUnsafeSteeringVectorLayer(self.steering_layer))
        
        new_module = MergedTransformerRepSteerLayer(transformer_layer=target, steering_layer=self.steering_layers[-1])
        setattr(parent, target_name, new_module)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        # for some reason we need to do this? not sure why
        model.enable_input_require_grads()

        for m in model.modules():
            if isinstance(m, TiedUnsafeSteeringVectorLayer):
                m.tied_module.rep_steer_directions.requires_grad = True

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        #self._set_adapter_layers(enabled=True)
        assert False

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        #for active_adapter in self.active_adapters:
        #    val = self.peft_config[active_adapter].bias
        #    if val != "none":
        #        msg = (
        #            f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
        #            "output as the the base model would without adaption."
        #        )
        #        warnings.warn(msg)
        #self._set_adapter_layers(enabled=False)
        assert False

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        print("rep_steer/_set_adapter_layers:")
        from IPython import embed; embed(); exit()

    @staticmethod
    def _check_target_module_exists(rep_steer_config, key):
        return check_target_module_exists(rep_steer_config, key)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, TiedUnsafeSteeringVectorLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)