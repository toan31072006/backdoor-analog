import numpy as np

from attack.tsba import build_tsba_trigger

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- kinematics
def velocity_profiles_from_skeleton(npy_path, sample_idx=None, top_k=6,
                                    fps=30.0, los=(1.0, 0.0, 0.0)):
    """Radial velocity time-series of the top-moving joints of one action instance.

    Accepts NTU-style (N,3,T,V,M) trigger skeletons (e.g. data_bend.npy).
    Returns vel_radial (J,Tf) in m/s, pos_radial (J,), used_joints (J,).
    """
    a = np.load(npy_path, allow_pickle=True)
    if a.ndim == 5:
        a = a[..., 0]                                  # person 1 -> (N,3,T,V)
    N = a.shape[0]
    if sample_idx is None:
        occ = (a != 0).reshape(N, -1).mean(1)
        sample_idx = int(np.argmax(occ))
    s = a[sample_idx].transpose(1, 2, 0)               # (T,V,3)
    valid = np.abs(s).sum((1, 2)) > 1e-6
    s = s[valid]
    los = np.asarray(los, float); los = los / np.linalg.norm(los)
    vel = np.diff(s, axis=0) * fps                     # (Tf,V,3) m/s
    speed = np.linalg.norm(vel, axis=2)
    movers = np.argsort(-speed.mean(0))[:top_k]
    vel_radial = (vel[:, movers, :] @ los).T           # (J,Tf)
    pos_radial = (s[:, movers, :] @ los).mean(0)       # (J,)
    return vel_radial, pos_radial, movers


class MicroDopplerTrigger:
    def __init__(self, n_ant=3, n_sub=30, n_pkt=20, fc=5.32e9, df=312.5e3,
                 packet_rate=1000.0, aoa_spread=0.6, seed=0, zero_mean=False):
        self.n_ant, self.n_sub, self.n_pkt = n_ant, n_sub, n_pkt
        self.fc, self.df, self.dt = fc, df, 1.0 / packet_rate
        self.lam = C_LIGHT / fc
        self.k = np.arange(n_sub)
        self.aoa_spread = aoa_spread
        self.rng = np.random.default_rng(seed)
        self.zero_mean = bool(zero_mean)   # MMFi branch: gain 1 + alpha*p with mean(p)=0
        self.m = None
        self.m_zm = None

    def build(self, vel_radial, pos_radial, d0=1.5):
        J, Tf = vel_radial.shape
        xq = np.linspace(0, 1, self.n_pkt)
        xp = np.linspace(0, 1, Tf)
        vel = np.stack([np.interp(xq, xp, vel_radial[j]) for j in range(J)])   # (J,P)
        nu = 2.0 * vel / self.lam
        phase_t = np.cumsum(2 * np.pi * nu * self.dt, axis=1)                  # (J,P)
        tau = 2.0 * (d0 + pos_radial) / C_LIGHT
        phase_f = -2 * np.pi * (self.k[None, :] * self.df) * tau[:, None]      # (J,S) ~flat
        gain = np.linalg.norm(vel, axis=1); gain = gain / (gain.sum() + 1e-12)
        aoa = self.rng.uniform(-self.aoa_spread, self.aoa_spread, size=(J, self.n_ant))
        m = np.zeros((self.n_ant, self.n_sub, self.n_pkt), complex)
        for a in range(self.n_ant):
            acc = np.zeros((self.n_sub, self.n_pkt), complex)
            for j in range(J):
                acc += gain[j] * np.exp(1j * phase_f[j])[:, None] \
                       * np.exp(1j * (phase_t[j] + aoa[j, a]))[None, :]
            m[a] = acc
        m = m / (np.sqrt((np.abs(m) ** 2).mean()) + 1e-12)
        self.m = m
        # Real-valued, zero-mean, unit-RMS projection for the DC-free MMFi gain.
        # The mean is removed from the *pattern* (before alpha), so
        # mean(1 + alpha*p) == 1 for every dose and RMS(alpha*p) == alpha.
        p = m.real - m.real.mean()
        self.m_zm = p / (np.sqrt((p ** 2).mean()) + 1e-12)
        return m

    def inject(self, csi, dose, eps=0.3):
        """
        Inject trigger into CSI frame.

        Supports two formats:
          Person-in-WiFi-3D : shape (3, 180, 20) — amp/phase layout with
                              3 link groups x 30 physical subcarriers per half
          MMFI               : shape (3, 114, 10) — float amplitude only in [0,1]
                              approximates complex injection as amplitude gain.
        """
        C, H, W = csi.shape
        complex_groups = 3
        complex_height = 2 * complex_groups * self.n_sub
        assert (C, H, W) == (self.n_ant, complex_height, self.n_pkt) or \
               (C, H, W) == (self.n_ant, self.n_sub, self.n_pkt), \
            f"Unexpected CSI shape {csi.shape} for trigger (n_ant={self.n_ant}, " \
            f"n_sub={self.n_sub}, n_pkt={self.n_pkt})"

        if H == complex_height:
            # Person-in-WiFi-3D: first half amplitude, second half phase.
            half = complex_groups * self.n_sub
            amp = csi[:, :half, :]
            ph  = csi[:, half:, :]
            A = amp.reshape(
                self.n_ant, complex_groups, self.n_sub, self.n_pkt)
            P = ph.reshape(
                self.n_ant, complex_groups, self.n_sub, self.n_pkt)
            H_cplx = A * np.exp(1j * P)
            m = self.m[:, None, :, :]
            Ht = H_cplx * (1.0 + dose * eps * m)
            At = np.abs(Ht).reshape(self.n_ant, half, self.n_pkt)
            Pt = np.angle(Ht).reshape(self.n_ant, half, self.n_pkt)
            out = np.concatenate([At, Pt], axis=1).astype(np.float32)
        else:
            # MMFI stores amplitude only.  Applying |1 + alpha*m| as a gain is
            # the amplitude-domain counterpart of the complex multiplication
            # used above.  It retains the signed phase/time variation of m;
            # adding |m| directly produced an almost-DC offset that the old
            # per-sample min-max normalization subsequently erased.
            alpha = float(dose) * float(eps)
            if self.zero_mean:
                # DC-free gain: |1+alpha*m| carries a positive bias (~alpha^2|m|^2/2)
                # plus mean(Re m) != 0, i.e. ~70% of its energy is a global
                # brightness offset that even a clean model reacts to.
                amp_gain = np.clip(1.0 + alpha * self.m_zm, 0.0, None).astype(np.float32)
            else:
                amp_gain = np.abs(1.0 + alpha * self.m).astype(np.float32)
            out = np.clip(csi.astype(np.float32) * amp_gain, 0.0, 1.0)
        return out


def load_trigger(action_npy, **kw):
    t = MicroDopplerTrigger(**{k: v for k, v in kw.items()
                               if k in MicroDopplerTrigger.__init__.__code__.co_varnames})
    vel, pos, movers = velocity_profiles_from_skeleton(
        action_npy, top_k=kw.get('top_k', 6))
    t.build(vel, pos)
    t.moving_joints = movers
    return t


def build_trigger_by_name(trigger_name: str, cfg: dict):
    """
    Factory: build a trigger by name from a config dict.

    trigger_name : micro_dropper, tsba, three traditional CSI adapters,
                   or an RF adapter (the latter requires its staged trainer).
    cfg          : the full experiment config dict

    Returns a trigger object with an .inject(csi, dose, eps) method.
    """
    name = trigger_name.lower().replace('-', '_')

    # ``n_sub`` is the number of physical subcarriers used to synthesize m.
    # MMFi exposes 114 amplitude bins directly. PiW3D stores 3x30 amplitudes
    # followed by 3x30 phases, so n_sub=30 (not the feature height 180).
    is_mmfi = cfg.get('experiment_name', '') == 'mmfi'
    n_sub_default = 114 if is_mmfi else 30
    n_pkt_default = 10  if is_mmfi else 20
    n_sub = cfg.get('n_sub', n_sub_default)
    n_pkt = cfg.get('n_pkt', n_pkt_default)
    n_ant = cfg.get('n_ant', 3)

    if name in ('micro_dropper', 'microdropper', 'micro_doppler'):
        t = MicroDopplerTrigger(
            n_ant=n_ant,
            n_sub=n_sub,
            n_pkt=n_pkt,
            aoa_spread=cfg.get('aoa_spread', 0.6),
            seed=cfg.get('seed', 0),
            zero_mean=cfg.get('trigger_zero_mean', False),
        )
        vel, pos, _ = velocity_profiles_from_skeleton(
            cfg['action_npy'], top_k=cfg.get('top_k', 6))
        t.build(vel, pos)
        return t

    if name in ('tsba', 'tsba_adapted'):
        return build_tsba_trigger(cfg)

    if name in ('badnets', 'blended', 'blend', 'wanet'):
        if cfg.get('experiment_name') != 'mmfi':
            raise ValueError('Traditional CSI adaptations are currently MM-Fi only')
        from attack.traditional import build_traditional_trigger
        return build_traditional_trigger(name, cfg)

    if name in ('infocom2025_por', 'ccai2026_backdoorrf'):
        from attack.rf_adapters import build_rf_trigger
        return build_rf_trigger(name, cfg)

    if name in ('sig', 'sig_adapter'):
        raise ValueError(
            f"Trigger '{trigger_name}' has been removed. SIG, Blended and WaNet are "
            "image-domain backdoors tuned for 0-255 pixels; on MMFi's [0,1] amplitude "
            "CSI they either swamp the signal (SIG: ~2400% mean perturbation, range "
            "[-19, 20]) or barely perturb it (WaNet: warp under one subcarrier cell). "
            "Re-derive their scales for [0,1] CSI before reinstating them.")

    raise ValueError(
        f"Unknown trigger '{trigger_name}'. Available triggers: "
        "'micro_dropper', 'tsba', 'badnets', 'blended', 'wanet', "
        "'infocom2025_por', 'ccai2026_backdoorrf'.")
