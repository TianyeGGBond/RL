"""RLix integration hooks for NeMo RL training loop.

Provides RLixHooks protocol and NoOpRLixHooks default for standalone mode.

Import direction: NeMo-side code (grpo.py, async_utils.py) imports this
module; the RLix pipeline provides the real implementation. One-way
dependency avoids circular imports between the two repos.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

# Derived once at import time; stable for the lifetime of the process.
DO_TIME_SHARING: bool = os.environ.get("RLIX_CONTROL_PLANE", "") == "rlix"


@runtime_checkable
class RLixHooks(Protocol):
    """Callbacks injected into async_grpo_train for RLix scheduler integration."""

    def before_training(self, step: int) -> None:
        """Block until the scheduler grants the training GPU allocation.

        The scheduler asynchronously shrinks overlap inference workers once
        the allocation request is filed, freeing VRAM for the training phase.
        Must be called before policy.prepare_for_training().
        """
        ...

    def after_training(self, step: int) -> None:
        """Notify the scheduler that the training GPU allocation is released.

        The scheduler asynchronously expands overlap inference workers and
        triggers selective weight sync via NemoRLModelUpdateService.
        Expand completion is guaranteed before the next before_training()
        call returns (enforced by the blocking request_gpus call).
        """
        ...

    def on_trajectory_collector_created(self, collector: Any) -> None:
        """Register the trajectory collector actor handle with the pipeline.

        Called once after the collector Ray actor is created in async_grpo_train.
        The pipeline actor stores the handle so _expand_workers can call
        set_weight_version on it after each selective sync completes.
        """
        ...


class NoOpRLixHooks:
    """No-op hooks for standalone (non-RLix) mode. All methods are pass-through."""

    def before_training(self, step: int) -> None:
        pass

    def after_training(self, step: int) -> None:
        pass

    def on_trajectory_collector_created(self, collector: Any) -> None:
        pass
