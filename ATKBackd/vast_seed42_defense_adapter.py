"""Explicit architecture ports of the frozen clean-data repair mechanisms.

No historical factory or repair source is changed. CLP pairs below follow the
actual pinned forward graphs, including ResNet non-Sequential conv/BN pairs.
Attention-similarity and concatenation norms are disclosed, not silently paired.
"""
from __future__ import annotations

import copy
from pathlib import Path

BACKBONES = ('metafiplusplus', 'wiflow', 'hpeli')
DEFENSES = ('none', 'ft', 'anp', 'nad', 'clp')
NAD_LAYERS = {
    'metafiplusplus': ('bn2', 'tf', 'decode.2'),
    'wiflow': ('residual_blocks.3', 'att', 'final_conv.2'),
    'hpeli': ('skunit1', 'skunit2', 'regression.5'),
}


def validate_selection(backbone, defense):
    if backbone not in BACKBONES or defense not in DEFENSES:
        raise ValueError('Only the declared MM-Fi backbone and none/FT/ANP/NAD/CLP are supported')


def recipe(backbone, defense):
    validate_selection(backbone, defense)
    if defense == 'anp':
        from defend.a1_anp import resolve_anp_recipe
        return resolve_anp_recipe()
    if defense in ('ft', 'nad'):
        from defend.a1_repair_ops import DEFAULT_RECIPE
        value = copy.deepcopy(DEFAULT_RECIPE)
        value['nad_layers'] = list(NAD_LAYERS[backbone])
        return value
    return dict(u=3.0) if defense == 'clp' else {}


def _pair_names(backbone):
    if backbone == 'metafiplusplus':
        pairs = [('encoder_conv1_p1', 'encoder_bn1_p1')]
        for stage, size in enumerate((3, 4, 6, 3), start=1):
            for block in range(size):
                prefix = f'encoder_layer{stage}_p1.{block}'
                pairs += [(prefix+'.conv1', prefix+'.bn1'), (prefix+'.conv2', prefix+'.bn2')]
                if stage > 1 and block == 0:
                    pairs.append((prefix+'.downsample.0', prefix+'.downsample.1'))
        pairs += [('tf.reconstruct_1.conv', 'tf.reconstruct_1.norm'),
                  ('decode.0', 'decode.1'), ('decode.3', 'decode.4')]
        return pairs, {'bn2': 'concatenation of three shared-encoder antenna features; no single preceding Conv2d'}
    if backbone == 'wiflow':
        pairs = []
        for prefix in ('up', *(f'residual_blocks.{i}' for i in range(4))):
            pairs += [(prefix+f'.block.{i}', prefix+f'.block.{i+1}') for i in (0,4,8)]
            pairs.append((prefix+'.downsample.0', prefix+'.downsample.1'))
        pairs.append(('final_conv.0', 'final_conv.1'))
        skips = {f'att.{axis}_axis.bn_similarity':
                 'normalizes QK similarity products, not a direct Conv2d channel transform'
                 for axis in ('width', 'height')}
        return pairs, skips
    if backbone == 'hpeli':
        # Retain the audited historical HPE-Li coverage, without modifying it.
        return None, None
    raise ValueError('Unknown backbone')


def clp_coverage(model, backbone):
    """Audit every BN2d against source-grounded pairs or a named semantic skip."""
    from torch import nn
    validate_selection(backbone, 'clp')
    if backbone == 'hpeli':
        from defend.a1_clp import _pairing, HPELI_UNPAIRED_BN2D
        pairs, norms, bn1d, skips = _pairing(model, HPELI_UNPAIRED_BN2D)
        return dict(backbone=backbone, pairs=[[p[0], p[2]] for p in pairs],
            paired_bn2d_count=len(pairs), total_bn2d_count=len(norms), skipped_bn2d=skips,
            unsupported_bn1d_names=bn1d, scope='audited historical direct Conv2d-BN2d HPE-Li adaptation')
    pairs, skip_reasons = _pair_names(backbone)
    norms = {n: m for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}
    aliases = [m for _, m in model.named_modules(remove_duplicate=False) if isinstance(m, nn.BatchNorm2d)]
    if len(aliases) != len({id(m) for m in aliases}):
        raise ValueError('CLP shared BN aliases are not supported')
    paired = []
    conv_ids = set()
    for conv_name, bn_name in pairs:
        try:
            conv, norm = model.get_submodule(conv_name), model.get_submodule(bn_name)
        except (AttributeError, KeyError) as exc:
            raise ValueError('Pinned CLP architecture names changed') from exc
        if (type(conv) is not nn.Conv2d or type(norm) is not nn.BatchNorm2d
                or conv.out_channels != norm.num_features or id(conv) in conv_ids):
            raise ValueError('Pinned CLP direct Conv2d-BN2d geometry changed')
        conv_ids.add(id(conv)); paired.append(bn_name)
    if len(set(paired)) != len(paired) or set(norms) != set(paired) | set(skip_reasons):
        raise ValueError('CLP BN2d coverage mismatch; no undeclared or silently skipped norms')
    skips = []
    for name, reason in skip_reasons.items():
        parent = model.get_submodule(name.rpartition('.')[0]) if '.' in name else model
        if backbone == 'wiflow' and type(parent).__name__ != 'AxialAttention':
            raise ValueError('CLP similarity-norm skip no longer belongs to axial attention')
        skips.append(dict(name=name, channels=norms[name].num_features, reason=reason))
    return dict(backbone=backbone, pairs=[list(p) for p in pairs],
        paired_bn2d_count=len(pairs), total_bn2d_count=len(norms), skipped_bn2d=skips,
        unsupported_bn1d_names=[n for n,m in model.named_modules() if isinstance(m, nn.BatchNorm1d)],
        scope='only source-audited direct Conv2d-BN2d pairs; BN1d and semantic skips untouched')


def prune_clp(model, backbone):
    """Fixed-u3, data-free rule; no clean fit, threshold search or post-prune FT."""
    import torch
    from defend.a1_clp import _channel_scores
    coverage = clp_coverage(model, backbone)
    layers = []
    for conv_name, bn_name in coverage['pairs']:
        conv, norm = model.get_submodule(conv_name), model.get_submodule(bn_name)
        scores, dtype = _channel_scores(conv, norm, bn_name)
        threshold = scores.mean() + 3.0*scores.std(correction=1)
        if not torch.isfinite(threshold):
            raise ValueError('Nonfinite CLP threshold')
        indices = torch.nonzero(scores > threshold, as_tuple=False).flatten().tolist()
        layers.append(dict(conv_name=conv_name, bn_name=bn_name, channels=conv.out_channels,
            score_dtype=dtype, scores=scores.tolist(), threshold=float(threshold),
            pruned_indices=indices, pruned_channels=len(indices)))
    repaired = copy.deepcopy(model)
    with torch.no_grad():
        for layer in layers:
            norm = repaired.get_submodule(layer['bn_name'])
            norm.weight[layer['pruned_indices']] = 0
            norm.bias[layer['pruned_indices']] = 0
    return repaired.eval(), dict(defense='clp', complete=True, u=3.0, coverage=coverage,
        layers=layers, pruned_channels=sum(r['pruned_channels'] for r in layers),
        uses_data=False, selection_on_test=False, optimizer_steps=0, fine_tuning_epochs=0,
        rule='sigma_max(kernel[Cin/groups,Kh*Kw])*abs(BN.gamma)/sqrt(BN.running_var); score>mean+3*sampleSD',
        std_correction=1, variance_epsilon_added=False)


def architecture_audit(model, backbone, defense):
    from torch import nn
    validate_selection(backbone, defense)
    result = dict(backbone=backbone, defense=defense)
    if defense == 'clp':
        result.update(clp_coverage(model, backbone))
    if defense == 'anp':
        norms = [(n,m) for n,m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not norms or any(type(m) not in (nn.BatchNorm1d,nn.BatchNorm2d) or not m.affine for _,m in norms):
            raise ValueError('ANP requires eligible ordinary affine BN1d/BN2d layers')
        result.update(masked_bn_names=[n for n,_ in norms],
            excluded_layernorm_names=[n for n,m in model.named_modules() if isinstance(m, nn.LayerNorm)],
            scope='ordinary BN1d/BN2d only; graph/attention LayerNorm untouched')
    if defense == 'nad':
        for name in NAD_LAYERS[backbone]:
            model.get_submodule(name)
        result.update(layers=list(NAD_LAYERS[backbone]), betas=[500.,1000.,1000.],
            scope='three fused 4D stage maps, never a shared per-antenna hook retaining only its final call')
    return result


def _repair_impl(model, backbone, defense, fit, selection, device, checkpoint, binding, seed=42):
    """Use unchanged pinned operators; only architecture mappings are supplied."""
    architecture = architecture_audit(model, backbone, defense)
    if defense == 'none':
        return model.eval(), dict(defense='none', complete=True, optimizer_steps=0,
                                 original_checkpoint_reused=True, architecture=architecture)
    if defense == 'clp':
        return prune_clp(model, backbone)
    if defense == 'anp':
        from defend.a1_anp import train_anp
        repaired, metadata = train_anp(model, fit, selection, recipe(backbone, defense), device,
                                      checkpoint, binding, seed)
    else:
        from defend.a1_repair_ops import train_repair
        repaired, metadata = train_repair(model, fit, selection, defense, recipe(backbone, defense),
                                         device, checkpoint, binding, seed)
    metadata = copy.deepcopy(metadata)
    metadata['architecture'] = architecture
    return repaired, metadata


def repair(model, backbone, defense, fit, selection, device, checkpoint, binding, seed=42,
           *, activation_offload='none'):
    from vast_seed42_memory import activation_context, describe
    # Enclose the native operators, including NAD attention and both ANP losses.
    # No historical operator/model/recipe is modified or forward recomputed.
    with activation_context(activation_offload, device):
        repaired, metadata = _repair_impl(model, backbone, defense, fit, selection, device,
                                          checkpoint, binding, seed)
    if activation_offload == 'cpu':
        metadata = copy.deepcopy(metadata)
        metadata['activation_memory'] = describe(activation_offload)
    return repaired, metadata


def _state_digest(value):
    """Lossless tensor/state fingerprint for the explicit small-batch parity gate."""
    import hashlib
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(('tensor:'+str(item.dtype)+':'+str(tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(('array:'+str(item.dtype)+':'+str(item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=str): visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update((type(item).__name__+str(len(item))).encode())
            for child in item: visit(child)
        else:
            digest.update((type(item).__name__+':'+repr(item)).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def smoke(backbone, defense, device='cpu', batch_size=2, hold=None, *, activation_offload='none',
          capture_state=False):
    from vast_seed42_memory import activation_context
    with activation_context(activation_offload, device):
        result = _smoke_impl(backbone, defense, device, batch_size, hold, capture_state=capture_state)
    if activation_offload == 'cpu': result['activation_offload'] = 'cpu'
    return result


def _smoke_impl(backbone, defense, device='cpu', batch_size=2, hold=None, *, capture_state=False):
    """Explicit synthetic backward/coverage smoke; not an efficacy experiment.

    Call with physical batch256 on the intended GPU to measure the memory gate.
    There is no automatic batch shrink, allocation fallback or production recipe change.
    """
    import torch
    from vast_seed42_adapter import build_model
    from defend.a1_repair_ops import attention_loss, pose_loss, pose_prediction
    from run_piw_hpeli_clean import _setup_execution
    validate_selection(backbone, defense)
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError('Smoke batch size must be at least two (BN training)')
    if hold is not None and not callable(hold):
        raise ValueError('Smoke hold must be an explicit callable or None')
    execution = _setup_execution(device)
    if device.startswith('cuda:'):
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(backbone,num_keypoints=17,num_coor=3,num_person=1,
                        subcarrier_num=114,dataset='mmfi',pretrained=False).to(device)
    audit = architecture_audit(model, backbone, defense)
    x, y = torch.randn(batch_size,3,114,10,device=device), torch.randn(batch_size,17,3,device=device)
    handles, features, teachers = [], {}, {}
    teacher = None
    optimizer = mask_optimizer = noise_optimizer = None
    optimizer_steps = 0
    optimizer_state_initialized = False
    gradient_trace, output_trace = [], []
    def track(network):
        if capture_state:
            for name, parameter in network.named_parameters():
                if parameter.requires_grad:
                    def capture_gradient(gradient, name=name):
                        gradient_trace.append((name, _state_digest(gradient)))
                    handles.append(parameter.register_hook(capture_gradient))
    try:
        if defense == 'nad':
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            for network, dest in ((model,features),(teacher,teachers)):
                for name in NAD_LAYERS[backbone]:
                    def capture(_module,_args,output,name=name,dest=dest):
                        if not isinstance(output,torch.Tensor) or output.ndim != 4 or len(output)!=batch_size or name in dest:
                            raise ValueError('NAD expected one fused 4D invocation per declared layer')
                        dest[name] = output
                    handles.append(network.get_submodule(name).register_forward_hook(capture))
        if defense == 'anp':
            from defend.a1_anp import masked_model, _mask_iteration
            working, names = masked_model(model)
            masks=[p for n,p in working.named_parameters() if n.endswith('.neuron_mask')]
            noises=[p for n,p in working.named_parameters() if n.endswith(('.neuron_noise','.neuron_noise_bias'))]
            r=recipe(backbone,defense)
            mask_optimizer=torch.optim.SGD(masks,lr=r['lr'],momentum=r['momentum'])
            noise_optimizer=torch.optim.SGD(noises,lr=r['anp_eps']/r['anp_steps'])
            track(working)
            for _ in range(2):
                value=_mask_iteration(working,x,y,r,mask_optimizer,noise_optimizer)
                if capture_state: output_trace.append(value)
                optimizer_steps+=1
            optimizer_state_initialized=bool(mask_optimizer.state)
            finite=bool(torch.isfinite(torch.tensor(value)))
        elif defense in ('none','clp'):
            fixed=model.eval() if defense=='none' else prune_clp(model,backbone)[0]
            with torch.no_grad():
                finite=bool(torch.isfinite(pose_prediction(fixed,x)).all())
        else:
            optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
            track(model)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                features.clear(); teachers.clear()
                prediction=pose_prediction(model.train(),x)
                loss=pose_loss(prediction,y)
                if teacher is not None:
                    with torch.no_grad(): pose_prediction(teacher,x)
                    loss=loss+sum(b*attention_loss(features[n],teachers[n]) for n,b in zip(NAD_LAYERS[backbone],(500.,1000.,1000.)))
                if capture_state: output_trace.append((_state_digest(prediction), float(loss.detach())))
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError('Nonfinite synthetic defense smoke')
                optimizer.step(); optimizer_steps+=1
            finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
            optimizer_state_initialized=bool(optimizer.state)
            if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                raise ValueError('NAD teacher unexpectedly received gradients')
        if not finite: raise RuntimeError('Nonfinite synthetic defense smoke')
        if device.startswith('cuda:'): torch.cuda.synchronize(device)
        if hold is not None: hold()
        result = dict(status='SYNTHETIC_DEFENSE_SMOKE_NOT_FULL_MMFI', backbone=backbone, defense=defense,
            physical_batch=batch_size, execution=execution, architecture=audit, finite=True,
            optimizer_steps=optimizer_steps,optimizer_state_initialized=optimizer_state_initialized,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda:') else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda:') else None)
        if capture_state:
            from run_piw_hpeli_clean import _rng_state
            final_model = working if defense == 'anp' else fixed if defense in ('none','clp') else model
            states = [item.state_dict() for item in (optimizer,mask_optimizer,noise_optimizer) if item is not None]
            result['parity_state'] = dict(outputs=_state_digest(output_trace), gradients=_state_digest(gradient_trace),
                model_and_bn=_state_digest(final_model.state_dict()), optimizer=_state_digest(states),
                teacher=_state_digest(teacher.state_dict() if teacher is not None else None),
                rng=_state_digest(_rng_state(device)))
        return result
    finally:
        for handle in handles: handle.remove()


def offload_parity(backbone, defense, device='cpu'):
    """Two-step native/offloaded comparison at B2; CUDA copies only on CUDA.

    RNG is restored around the diagnostic. This is not a batch256 capacity test.
    """
    import torch
    from run_piw_hpeli_clean import _rng_state, _restore_rng
    if defense not in ('ft','nad','anp'):
        raise ValueError('Offload parity applies to trainable repairs only')
    saved = _rng_state(device)
    saved_cuda_all = torch.cuda.get_rng_state_all() if device.startswith('cuda:') else None
    try:
        torch.manual_seed(42)
        if device.startswith('cuda:'): torch.cuda.manual_seed(42)
        initial = _rng_state(device)
        native = smoke(backbone, defense, device, 2, capture_state=True)['parity_state']
        _restore_rng(initial, device)
        offloaded = smoke(backbone, defense, device, 2, activation_offload='cpu', capture_state=True)['parity_state']
        if native != offloaded:
            changed = [key for key in native if native[key] != offloaded[key]]
            raise RuntimeError('CPU-offload numerical parity failed: '+','.join(changed))
        return dict(complete=True, physical_batch=2, optimizer_steps=2,
                    device=device, actual_cuda_copies=device.startswith('cuda:'), checks=native,
                    limitation='small-batch parity only, not physical256 capacity or full-data validation')
    finally:
        _restore_rng(saved, device)
        if saved_cuda_all is not None: torch.cuda.set_rng_state_all(saved_cuda_all)
