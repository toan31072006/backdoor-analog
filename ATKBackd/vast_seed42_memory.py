"""Explicit activation-storage policy; never changes a scientific recipe.

``cpu`` uses PyTorch's official saved-tensor offload context. Forward execution,
physical batch, BatchNorm calls, precision and optimizer steps stay unchanged.
It does not recompute forward or fall back to another batch/precision. CPU-only
tests do not verify CUDA copies, actual peak memory, throughput or host RAM fit.
"""
from __future__ import annotations

from contextlib import nullcontext
import re

POLICIES = ('none', 'cpu')


def validate_policy(policy):
    """Reject implicit, misspelled or coerced policies before doing any work."""
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError('Activation offload policy must be explicitly none or cpu')
    return policy


def activation_context(policy, device):
    """Create an explicit no-op or CPU saved-autograd-tensor context.

    The official context copies saved tensors to CPU, then restores them to
    their original device when backward needs them. ``pin_memory=False`` avoids
    silently reserving pinned host RAM. With CPU inputs it remains a CPU no-op;
    that path is useful for local correctness tests, not GPU-memory validation.
    The context must cover production and smoke forward/loss construction alike.
    """
    policy = validate_policy(policy)
    name = str(device)
    if re.fullmatch(r'(?:cpu|cuda:(?:0|[1-9][0-9]*))', name) is None:
        raise ValueError('Activation storage requires cpu or an explicit cuda:N device')
    if policy == 'none':
        return nullcontext()
    import torch
    return torch.autograd.graph.save_on_cpu(pin_memory=False, device_type='cuda')


def describe(policy):
    """JSON-safe execution metadata; not a claim of measured memory savings."""
    policy = validate_policy(policy)
    return dict(policy=policy,
        mechanism='none' if policy == 'none' else 'torch.autograd.graph.save_on_cpu',
        pin_memory=False, forward_recomputation=False,
        physical_batch_changed=False, precision_changed=False,
        validation_note='CPU-only tests are not actual CUDA/offload/RAM/throughput validation')
