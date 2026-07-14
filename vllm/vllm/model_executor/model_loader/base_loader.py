# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
import importlib
import os

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


def _maybe_import_vllm_tool():
    value = os.environ.get("VLLM_IPC_TOOL", "")
    if value.lower() not in {"1", "true", "yes", "on"}:
        return None
    import sys

    ipc_tools_path = os.environ.get(
        "VLLM_IPC_TOOLS_PATH",
        next((str(_p / "IPC" / "ipc_tools")
              for _p in __import__("pathlib").Path(__file__).resolve().parents
              if (_p / "IPC" / "ipc_tools").is_dir()), ""),
    )
    if ipc_tools_path not in sys.path:
        sys.path.insert(0, ipc_tools_path)
    module_name = os.environ.get("VLLM_IPC_TOOL_MODULE", "vllm_tool")
    return importlib.import_module(module_name)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        vllm_tool = _maybe_import_vllm_tool()

        with set_default_torch_dtype(model_config.dtype):
            if vllm_tool is not None and vllm_tool.ipc_enabled():
                assert vllm_tool.TOOL is not None
                assert vllm_tool.disguise_switch is True
                assert vllm_tool.disguise_kv is True
                with vllm_tool.TOOL.scope(name="Weight"):
                    with target_device:
                        model = initialize_model(
                            vllm_config=vllm_config, model_config=model_config
                        )
            else:
                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config, model_config=model_config
                    )

            log_model_inspection(model)

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            if vllm_tool is None or not vllm_tool.ipc_enabled() or vllm_tool.TOOL.is_server_creator:
                self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))
