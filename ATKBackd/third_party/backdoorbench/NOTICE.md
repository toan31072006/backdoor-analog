# BackdoorBench trigger operators

Copyright (c) 2022, CUHK(SZ), SRIBD. All rights reserved.

Upstream: https://github.com/SCLBD/BackdoorBench
Pinned commit: `f02e3534645f0ee63d6848653062cd6c0d6c400d`.

These files are excerpts of the **BackdoorBench implementation**, not releases by
BadNets/Blended's original-paper authors. They are distributed under the upstream
CC BY-NC 4.0 license; retain this notice and [LICENSE](LICENSE). Non-commercial
use only under that license. Provided as-is, without warranties; no endorsement
by upstream is implied. License text is also available at
https://creativecommons.org/licenses/by-nc/4.0/legalcode.

## Included source and modifications

- `patch.py`: upstream `utils/bd_img_transform/patch.py`, only the
  `AddMaskPatchTrigger` class. Unused imports and its optional type annotation
  are omitted. Its constructor, call interface and replacement arithmetic are
  retained: zero trigger entries are transparent; positive entries replace input.
- `blended.py`: upstream `utils/bd_img_transform/blended.py`; the class and
  arithmetic are unchanged.
- `LICENSE`: full upstream license notice and license texts.
- `SOURCE_MANIFEST.json`: immutable upstream URLs and SHA-256 hashes of the
  complete downloaded upstream files (not of the reduced local excerpt).

New attribution headers and whitespace normalization are the only other edits
in these vendored operator files. No remote code is downloaded at training time.

## MM-Fi adapter boundary

`attack/traditional.py` calls these operators on floating-point HWC arrays,
transposed from antenna/subcarrier/packet CSI. It deliberately does not use
BackdoorBench's PIL/uint8 image wrapper, which would quantize CSI.

BadNets uses a normalized white rectangular 8-by-3 patch. At dose1 the default
opacity1 directly replaces the patch; at smaller doses we interpolate toward
that replacement. Shared epsilon is not its patch opacity. Blended uses a fixed
seeded unit-range random pattern and alpha=dose*epsilon (default0.185), not
BackdoorBench's default Hello Kitty image and alpha0.2. Poison selection, the HPELi
model, MPJPE loss and per-sample pose payload are local task adaptations, not
BackdoorBench's categorical training pipeline.

The operators are source-backed; the complete experiment is still a CSI/HPE
adaptation, not an exact reproduction of the original image-domain experiments.
