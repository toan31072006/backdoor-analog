import os
import re
import numpy as np
from torch.utils.data import Dataset


def _read_list(path):
    with open(path, encoding='utf-8') as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


def _next_piw_frame_name(name):
    """DT-Pose naming rule used to filter its PiW3D training split."""
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f'unexpected Person-in-WiFi-3D frame name: {name!r}')
    try:
        frame_idx = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f'unexpected Person-in-WiFi-3D frame index in {name!r}') from exc
    return f'{parts[0]}_{parts[1]}_{frame_idx + 1}'


class PersonInWiFi3D(Dataset):
    def __init__(self, split, data_root, experiment_name='one-person', num_person=1):
        self.split = split
        self.num_person = num_person
        sub = 'train_data' if split == 'training' else 'test_data'
        self.root = os.path.normpath(os.path.join(data_root, sub))
        lst = os.path.join(self.root, f'{sub}_list.txt')
        names = _read_list(lst)
        selected = []
        for nm in names:
            try:
                pc = int(nm.split('_')[0][2])
            except (IndexError, ValueError):
                pc = 1
            keep = ({'one-person': 1, 'two-person': 2, 'three-person': 3}
                    .get(experiment_name, None))
            if keep is not None and pc != keep:
                continue
            selected.append(nm)

        # Match DT-Pose exactly: its training loader retains a frame only when
        # the next frame exists (the next frame is used by temporal victims,
        # even though HPE-Li itself consumes only the current frame).
        if split == 'training':
            selected_set = set(selected)
            selected = [nm for nm in selected
                        if _next_piw_frame_name(nm) in selected_set]

        self.items = []
        for nm in selected:
            csi_path = os.path.normpath(
                os.path.join(self.root, 'csi_ap', nm + '.npy'))
            kpt_path = os.path.normpath(
                os.path.join(self.root, 'keypoint', nm + '.npy'))
            if not os.path.exists(csi_path) or not os.path.exists(kpt_path):
                raise FileNotFoundError(
                    f'listed PiW3D sample is incomplete: {nm!r} '
                    f'(csi={csi_path}, keypoint={kpt_path})')
            self.items.append({
                'csi': csi_path,
                'kpt': kpt_path,
                'name': nm,
            })

    # ----- split the original read_frame into load + normalize -----------------------
    @staticmethod
    def load_raw(csi_path):
        return np.load(csi_path).astype(np.float32)

    @staticmethod
    def normalize(raw):
        amp = raw[:, :90, :]; ph = raw[:, 90:, :]
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
        ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-12)
        return np.concatenate([amp, ph], axis=1).astype(np.float32)

    def load_pose(self, kpt_path):
        p = np.load(kpt_path).astype(np.float32)
        if p.ndim == 2:                                  # (14,3) -> (1,14,3)
            p = p[None]
        if p.shape[0] < self.num_person:
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)
        pose = self.load_pose(it['kpt'])
        return {'csi': csi, 'pose': pose, 'name': it['name']}


# MMFI action protocols, verbatim from DT-Pose (feeder/mmfi.py :: decode_config).
_MMFI_ALL_SUBJECTS = [f'S{i:02d}' for i in range(1, 41)]
_MMFI_PROTOCOL_ACTIONS = {
    'protocol1': ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17',
                  'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27'],   # daily
    'protocol2': ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11',
                  'A12', 'A15', 'A16', 'A24', 'A25', 'A26'],          # rehabilitation
    'all': [f'A{i:02d}' for i in range(1, 28)],
}
# DT-Pose's paper calls the 27-action set P3; accept that name too so a config
# can say protocol3 instead of having to know it is spelled "all" here.
_MMFI_PROTOCOL_ACTIONS['protocol3'] = _MMFI_PROTOCOL_ACTIONS['all']


# S3 (cross-environment), from DT-Pose config/mmfi/pose_config.yaml ::
# cross_scene_split. Train sees E01-E03 with S01-S30; validation is E04 with
# S31-S40, so neither the environment nor the subject is shared.
#
# In MMFi each subject is recorded in exactly one environment (DT-Pose hard-codes
# this in feeder/mmfi.py :: get_scene), so the split is fully determined by the
# subject: taking the Cartesian product of scenes x subjects would triple-count
# every training key against environments that subject never appears in.
_MMFI_SUBJECT_SCENE = {f'S{i:02d}': f'E{(i - 1) // 10 + 1:02d}'
                       for i in range(1, 41)}          # S01-S10 -> E01, ... S31-S40 -> E04
_MMFI_S3_TRAIN_SUBJECTS = [f'S{i:02d}' for i in range(1, 31)]   # E01-E03
_MMFI_S3_VAL_SUBJECTS = [f'S{i:02d}' for i in range(31, 41)]    # E04


def _mmfi_scene_split(protocol):
    """DT-Pose's cross_scene_split: hold out a whole environment.

    Unlike S1, nothing is randomised — the partition is fixed, so the model is
    tested on an environment AND a subject it never saw.

    Returns (train_keys, val_keys) as sets of (scene, subject, action), each key
    using the subject's own environment. With the full dataset that is
    30 subjects x |actions| training sequences and 10 x |actions| validation.
    """
    actions = _MMFI_PROTOCOL_ACTIONS.get(protocol, _MMFI_PROTOCOL_ACTIONS['all'])
    train = {(_MMFI_SUBJECT_SCENE[s], s, a)
             for s in _MMFI_S3_TRAIN_SUBJECTS for a in actions}
    val = {(_MMFI_SUBJECT_SCENE[s], s, a)
           for s in _MMFI_S3_VAL_SUBJECTS for a in actions}
    return train, val


def _mmfi_subject_split(protocol, random_ratio, random_seed):
    """Reproduce DT-Pose's per-action subject split exactly.

    DT-Pose reseeds numpy for EVERY action (rs, rs+1, ...) and draws a fresh
    32/8 subject permutation each time, so a subject can train on one action
    and be held out on another. Shuffling whole sequences once instead — as
    this feeder used to — leaks ~80% of DT-Pose's test sequences into training
    and makes the numbers incomparable.

    Returns (train_pairs, val_pairs) as sets of (subject, action).
    """
    actions = _MMFI_PROTOCOL_ACTIONS.get(protocol, _MMFI_PROTOCOL_ACTIONS['all'])
    train_pairs, val_pairs = set(), set()
    rs = random_seed
    n_train = int(np.floor(random_ratio * len(_MMFI_ALL_SUBJECTS)))
    for action in actions:
        np.random.seed(rs)                       # DT-Pose uses the legacy global RNG
        idx = np.random.permutation(len(_MMFI_ALL_SUBJECTS))
        subs = np.array(_MMFI_ALL_SUBJECTS)
        for s in subs[idx[:n_train]]:
            train_pairs.add((str(s), action))
        for s in subs[idx[n_train:]]:
            val_pairs.add((str(s), action))
        rs += 1
    return train_pairs, val_pairs


def _build_mmfi_items(data_root, split,
                      protocol='protocol1',
                      setting='s1',
                      random_ratio=0.8,
                      random_seed=0):
    """
    Walk the MMFI Compress/ hierarchy and return a flat list of frame-level items.

    Directory structure (verified on local data):
        <data_root>/E##/S##/A##/
            ground_truth.npy        shape (T, 17, 3) — T frames for the sequence
            wifi-csi/
                frame001_processed.npy   shape (3, 114, 10) — already in [0,1]
                frame002_processed.npy
                ...

    Each item maps one CSI frame to its corresponding ground_truth row.

    Protocol x setting
    ------------------
    DT-Pose names a run ``protocol<N>-s<M>``: the PROTOCOL selects which actions
    are in scope, the SETTING selects how subjects and environments are split.
    The two are independent, so e.g. protocol1-s3 is a valid combination.

    protocol (which actions)
        protocol1 (default) — P1, 14 daily actions.
        protocol2           — P2, 13 rehabilitation actions (disjoint from P1).
        protocol3 / all     — P3, all 27 actions. Both spellings work:
                              "all" is this module's original name, "protocol3"
                              matches how the paper refers to it.
        random_split        — legacy escape hatch, NOT a protocol: one shuffle
                              over every sequence, ignoring the action subset.
                              Incomparable with DT-Pose (it leaks most of
                              DT-Pose's test sequences into training); kept only
                              for backwards compatibility. Ignores `setting`.

    setting (how it is split)
        s1 (default) — DT-Pose's random_split: for EACH action an independent
                       32/8 subject draw (seeds rs, rs+1, ...). Subjects recur
                       on both sides across actions, so this is the easier
                       split. protocol1-s1 gives 448 train / 112 val sequences.
        s3           — DT-Pose's cross_scene_split: train on E01-E03 (S01-S30),
                       validate on E04 (S31-S40). Neither environment nor
                       subject is shared, so it is the harder split.
                       protocol1-s3 gives 420 train / 140 val sequences.

        s2 (cross_subject_split) exists in DT-Pose but is NOT ported here and is
        rejected explicitly rather than falling back to s1.

    Note these are SEQUENCE counts; the number of frame-level samples returned
    is larger and depends on the sequence lengths actually present on disk.

    Args:
        data_root : path to MMFI/Compress/
        split     : 'training' | 'validation' | 'test'
                    'validation' and 'test' are treated identically.
        protocol  : 'protocol1' | 'protocol2' | 'protocol3' | 'all'
                    | 'random_split' (legacy).
        setting   : 's1' (default) | 's3'.
        random_ratio : train fraction, s1 only (default 0.8).
        random_seed  : base RNG seed, s1 only (default 0; s3 is deterministic).

    Returns:
        list of dicts with keys:
            'csi'  : absolute path to frame###_processed.npy
            'kpt'  : absolute path to ground_truth.npy  (shared by all frames in seq)
            'frame_idx' : int — row index into ground_truth.npy for this frame
            'name' : human-readable identifier  "E##_S##_A##_f####"
    """
    # Reject anything we do not actually implement. Without this, setting='s2'
    # would fall through to the s1 branch and silently run a random split while
    # the config, the logs and the checkpoint all claim cross-subject.
    if setting not in ('s1', 's3'):
        raise ValueError(
            f'unsupported MMFi setting {setting!r}. Implemented: "s1" '
            f'(random_split) and "s3" (cross_scene_split). DT-Pose also defines '
            f'"s2" (cross_subject_split), which is not ported here — do not pass '
            f'it expecting a cross-subject split.')

    root = os.path.normpath(data_root)

    # ── Enumerate all valid (env, subject, action) sequences ──────────────
    sequences = []   # list of (env, subj, action, seq_dir)
    for env in sorted(os.listdir(root)):
        env_dir = os.path.join(root, env)
        if not os.path.isdir(env_dir) or not env.startswith('E'):
            continue
        for subj in sorted(os.listdir(env_dir)):
            subj_dir = os.path.join(env_dir, subj)
            if not os.path.isdir(subj_dir) or not subj.startswith('S'):
                continue
            for action in sorted(os.listdir(subj_dir)):
                seq_dir = os.path.join(subj_dir, action)
                if not os.path.isdir(seq_dir) or not action.startswith('A'):
                    continue
                gt_path = os.path.join(seq_dir, 'ground_truth.npy')
                wifi_dir = os.path.join(seq_dir, 'wifi-csi')
                if os.path.exists(gt_path) and os.path.exists(wifi_dir):
                    sequences.append((env, subj, action, seq_dir))

    if not sequences:
        raise RuntimeError(
            f'No valid MMFI sequences found under {data_root}. '
            'Expected structure: <root>/E##/S##/A##/ground_truth.npy + wifi-csi/')

    # ── Train / val split ──────────────────────────────────────────────────
    # DT-Pose names these protocol<N>-s<M>: the protocol picks the action subset
    # (P1 daily / P2 rehab / P3 all), the setting picks how subjects and scenes
    # are partitioned (s1 random_split, s3 cross_scene_split).
    if setting == 's3':
        if protocol not in _MMFI_PROTOCOL_ACTIONS:
            raise ValueError(f'setting "s3" needs a protocol, got {protocol!r}')
        train_keys, val_keys = _mmfi_scene_split(protocol)
        keep = train_keys if split == 'training' else val_keys
        chosen_idx = {i for i, (env, subj, action, _) in enumerate(sequences)
                      if (env, subj, action) in keep}
    elif protocol in _MMFI_PROTOCOL_ACTIONS:
        train_pairs, val_pairs = _mmfi_subject_split(protocol, random_ratio, random_seed)
        keep = train_pairs if split == 'training' else val_pairs
        chosen_idx = {i for i, (_, subj, action, _) in enumerate(sequences)
                      if (subj, action) in keep}
    elif protocol == 'random_split':
        # Legacy behaviour, kept only for backwards compatibility. NOT comparable
        # to DT-Pose: it ignores the action protocol and shuffles all sequences.
        rng = np.random.default_rng(random_seed)
        idx = np.arange(len(sequences))
        rng.shuffle(idx)
        n_train = int(len(idx) * random_ratio)
        chosen_idx = set((idx[:n_train] if split == 'training' else idx[n_train:]).tolist())
    else:
        raise ValueError(f'unknown MMFI protocol {protocol!r}; expected one of '
                         f'{sorted(_MMFI_PROTOCOL_ACTIONS)} or "random_split"')

    # ── Build frame-level item list ────────────────────────────────────────
    items = []
    for seq_i, (env, subj, action, seq_dir) in enumerate(sequences):
        if seq_i not in chosen_idx:
            continue

        gt_path  = os.path.join(seq_dir, 'ground_truth.npy')
        wifi_dir = os.path.join(seq_dir, 'wifi-csi')

        # Load GT to know the total number of frames then release mmap handle
        gt_tmp = np.load(gt_path, mmap_mode='r')   # (T, 17, 3)
        n_frames = int(gt_tmp.shape[0])
        del gt_tmp   # release mmap file handle immediately — avoid fd exhaustion

        frame_files = sorted(
            f for f in os.listdir(wifi_dir) if f.endswith('_processed.npy')
        )

        # Pair each CSI frame with its ground_truth row by the frame NUMBER parsed
        # from the filename, not by position in the listing: a single missing
        # frame would otherwise shift every later frame against its pose.
        for fname in frame_files:
            m = re.search(r'frame(\d+)', fname)
            if m is None:
                continue
            fi = int(m.group(1)) - 1              # frame001 -> row 0
            if not (0 <= fi < n_frames):
                continue                          # no pose for this frame — skip
            items.append({
                'csi':       os.path.normpath(os.path.join(wifi_dir, fname)),
                'kpt':       os.path.normpath(gt_path),
                'frame_idx': fi,
                'name':      f'{env}_{subj}_{action}_f{fi+1:04d}',
            })

    return items


class MMFI(Dataset):
    """
    MMFI WiFi-CSI → 3-D pose dataset (17 COCO keypoints).

    Data layout (local copy at data_root = MMFI/Compress/):
        E##/S##/A##/
            ground_truth.npy     (T, 17, 3)  float32  — 3-D keypoints (meters)
            wifi-csi/
                frame###_processed.npy   (3, 114, 10)  float64 in [0, 1]

    The split is chosen by (protocol, setting), following DT-Pose's
    ``protocol<N>-s<M>`` naming. The default is protocol1-s1: 14 daily actions
    with an independent 32/8 subject draw per action (see _mmfi_subject_split).
    Pass setting='s3' for the harder cross-environment split, where training
    sees E01-E03/S01-S30 and validation is E04/S31-S40 (see _mmfi_scene_split).
    See _build_mmfi_items for the full protocol/setting matrix.

    CSI normalization:
        The raw data is already in [0, 1] (pre-processed by the MMFI authors).
        normalize() only validates, clips numerical overshoot, and casts to
        float32. It never rescales a sample, because doing so erases the
        amplitude-only trigger.

    Args:
        split     : 'training' | 'validation' | 'test'
        data_root : path to MMFI/Compress/ directory
        num_person: always 1 for MMFI (single-person dataset)
        protocol  : 'protocol1' (default, P1 daily) | 'protocol2' (P2 rehab)
                    | 'protocol3' / 'all' (P3, 27 actions) | 'random_split'
                    (legacy, not comparable with DT-Pose)
        setting   : 's1' (default, random_split) | 's3' (cross_scene_split).
                    's2' is refused — it is not ported.
        random_ratio : train fraction, s1 only (default 0.8 — DT-Pose ratio)
        random_seed  : base RNG seed, s1 only (default 0 — DT-Pose reseeds
                    per action); s3 is deterministic and ignores both
    """

    def __init__(self, split, data_root, num_person=1,
                 protocol='protocol1',
                 setting='s1',
                 random_ratio=0.8,
                 random_seed=0):
        self.split = split
        self.num_person = num_person
        self.data_root = os.path.normpath(data_root)

        # ``ground_truth.npy`` is shared by every frame in a sequence.  Keep a
        # process-local float32 copy so a persistent DataLoader worker opens it
        # once, not once per frame per epoch.  The source is opened as a mmap in
        # ``_load_ground_truth`` and closed immediately after the copy, which
        # avoids retaining hundreds of file descriptors per worker.
        self._ground_truth_cache = {}
        self._ground_truth_cache_pid = os.getpid()

        # Evaluation traverses the same split once clean and once per dose.
        # This cache is opt-in (see ``enable_evaluation_cache``), because the
        # training split is too large to retain frame-by-frame in RAM.
        self._raw_cache = None
        self._raw_cache_pid = None
        self._raw_cache_bytes = 0
        self._raw_cache_max_bytes = 0

        self.items = _build_mmfi_items(
            data_root=self.data_root,
            split=split,
            protocol=protocol,
            setting=setting,
            random_ratio=random_ratio,
            random_seed=random_seed,
        )

        if not self.items:
            raise RuntimeError(
                f'MMFI: no items found for split="{split}" under {self.data_root}. '
                'Check data_root and split name.')

    # ------------------------------------------------------------------
    def _reset_process_local_caches(self):
        """Drop inherited cache entries after fork/spawn process changes."""
        pid = os.getpid()
        if self._ground_truth_cache_pid != pid:
            self._ground_truth_cache = {}
            self._ground_truth_cache_pid = pid
        if self._raw_cache is not None and self._raw_cache_pid != pid:
            self._raw_cache = {}
            self._raw_cache_pid = pid
            self._raw_cache_bytes = 0

    def __getstate__(self):
        """Never pickle cached arrays into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state['_ground_truth_cache'] = {}
        state['_ground_truth_cache_pid'] = None
        if state.get('_raw_cache') is not None:
            state['_raw_cache'] = {}
            state['_raw_cache_pid'] = None
            state['_raw_cache_bytes'] = 0
        return state

    def enable_evaluation_cache(self, max_bytes=512 * 1024 * 1024):
        """Cache MM-Fi CSI frames across repeated evaluation passes.

        The canonical MM-Fi test split is about 434 MiB as float32.  Refuse to
        enable the cache when the documented frame shape would exceed the RAM
        budget; a partial same-order cache would thrash and provide no benefit.
        This changes I/O only: callers receive the exact same float32 arrays as
        ``np.load(...).astype(np.float32)``.

        Returns ``True`` when caching was enabled, otherwise ``False``.
        """
        max_bytes = int(max_bytes)
        if max_bytes < 0:
            raise ValueError('max_bytes must be non-negative')
        expected = (len(self.items) * 3 * 114 * 10
                    * np.dtype(np.float32).itemsize)
        if expected > max_bytes:
            self.clear_evaluation_cache()
            return False
        self._raw_cache = {}
        self._raw_cache_pid = os.getpid()
        self._raw_cache_bytes = 0
        self._raw_cache_max_bytes = max_bytes
        return True

    def clear_evaluation_cache(self):
        """Release cached CSI frames while leaving the pose cache intact."""
        self._raw_cache = None
        self._raw_cache_pid = None
        self._raw_cache_bytes = 0
        self._raw_cache_max_bytes = 0

    def load_raw(self, csi_path):
        """Load raw CSI frame.  Shape: (3, 114, 10), values in [0, 1]."""
        self._reset_process_local_caches()
        if self._raw_cache is not None:
            cached = self._raw_cache.get(csi_path)
            if cached is not None:
                return cached

        raw = np.load(csi_path, allow_pickle=False).astype(np.float32)
        if self._raw_cache is not None:
            next_bytes = self._raw_cache_bytes + raw.nbytes
            if next_bytes <= self._raw_cache_max_bytes:
                # All poisoning/normalization paths allocate their output.  A
                # read-only cache both documents and enforces that contract.
                raw.setflags(write=False)
                self._raw_cache[csi_path] = raw
                self._raw_cache_bytes = next_bytes
            else:
                # Shape/data drift made the preflight estimate too small.  Do
                # not retain a partial cache: sequential evaluation passes
                # would evict/reload every sample and only waste memory.
                self.clear_evaluation_cache()
        return raw

    @staticmethod
    def normalize(raw):
        """Return the authors' already-normalized MMFi frame unchanged.

        Re-running per-sample min-max normalization after poisoning removed
        the near-DC component of the amplitude-only MicroDoppler trigger.  The
        ``*_processed.npy`` files are already normalized to [0, 1].  Clipping
        is only a range safeguard; unlike min-max it does not rescale or erase
        the trigger pattern.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError('MMFI CSI contains NaN or Inf')
        return np.clip(raw, 0.0, 1.0)

    def _load_ground_truth(self, kpt_path):
        """Return one process-local, descriptor-free sequence pose array."""
        self._reset_process_local_caches()
        gt = self._ground_truth_cache.get(kpt_path)
        if gt is not None:
            return gt

        mapped = np.load(kpt_path, mmap_mode='r', allow_pickle=False)
        try:
            # Always copy, even when the file is already float32: otherwise an
            # ndarray view could keep the mmap (and its fd) alive in the cache.
            gt = mapped.astype(np.float32, copy=True)
        finally:
            mmap_obj = getattr(mapped, '_mmap', None)
            if mmap_obj is not None:
                mmap_obj.close()
        gt.setflags(write=False)
        self._ground_truth_cache[kpt_path] = gt
        return gt

    def load_pose(self, kpt_path, frame_idx):
        """
        Load the 3-D pose for a single frame.

        ground_truth.npy has shape (T, 17, 3).  We index row `frame_idx`
        to get the (17, 3) keypoints for this specific frame, then wrap
        it as (num_person, 17, 3).
        """
        gt = self._load_ground_truth(kpt_path)         # (T, 17, 3), cached per worker
        p  = gt[frame_idx].astype(np.float32)          # (17, 3), writable copy
        p  = p[None]                                   # (1, 17, 3)
        # Pad if num_person > 1 (not expected for MMFI, kept for API parity)
        if p.shape[0] < self.num_person:
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]                     # (num_person, 17, 3)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it  = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)                      # (3, 114, 10) float32
        pose = self.load_pose(it['kpt'], it['frame_idx'])  # (1, 17, 3) float32
        return {'csi': csi, 'pose': pose, 'name': it['name']}
