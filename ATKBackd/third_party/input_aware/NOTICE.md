# Pinned Input-Aware Dynamic Backdoor reference

Reference: Anh Nguyen and Anh Tran, **Input-Aware Dynamic Backdoor Attack**,
NeurIPS 2020. Official repository:
https://github.com/VinAIResearch/input-aware-backdoor-attack-release

Pinned commit: `1750e4ff0cc2371b243f1fd0ee18971f7e49fe78`.
The files in `upstream/` are unmodified references downloaded at this commit.
They are not imported by training or installed as a dependency. The MIT license
and copyright notice are retained in `upstream/LICENSE`.

| Relative upstream path | SHA-256 |
| --- | --- |
| train.py | fc375dc2a6e2254a0d05bd9736fb50c7ab663490ddb2f5c81ff43b021e359353 |
| config.py | 6bb7052023eca319ffabb1533ab37553a78d6c821bca0bbc7cb78a5fcf0729fd |
| networks/models.py | 92fc47dce6c8fc085be1c10b89a764750f939923087594805e4a5b8f34f844ca |
| networks/blocks.py | 8b456a0422ef86f21f8cf58dd61cb120aef716324e507a4d30f8c0b5f05c165d |
| LICENSE | 7d8600c83754d8766cfb8a3e4c0133bae45069beb598b4895f761fc9dfd139b3 |

## CSI/HPE adaptation, not original classification reproduction

`attack/input_aware_mmfi.py` independently instantiates the source architecture
and formulas. Tests load the pinned generator classes without torchvision to
check shape-padded network parity, threshold/blend parity, objective parity,
source defaults, and exact zero-dose identity.

Kept: two separate convolutional pattern/mask generators, widths32/64/128,
convolution/BN/ReLU/maxpool/bilinear upsample blocks, tanh pattern mapping,
thresholded one-channel mask, 25 mask warm-up epochs, mask-density/norm/diversity
objective, Adam optimizers and schedulers, frozen mask, joint victim/pattern
training, and the independently generated clean-labelled cross-trigger branch.

Changed and disclosed before running: CSI3x114x10 is bottom/right replicate
padded to120x16 and cropped back, not resized or min-max normalized; labels and
task loss are full-pose targets and MPJPE rather than class labels/cross-entropy.
Victim recipe remains the shared MM-Fi HPE-Li50-epoch recipe rather than the
source600-iteration image-classifier recipe. Mask warm-up25epochs is **additional
attacker optimization**, not victim epochs. TRAIN-only, predeclared poison and
cross identities/doses replace the source per-batch slots. Each poison has an
independently shuffled TRAIN donor for diversity, allowing unequal poison/cross
counts. Neither donors nor tuning may use TEST. There is no second normalization
or extra clipping/peak-budget projection.

Two implementation corrections are explicit: both victim and generator gradients
are zeroed every joint step (upstream omits generator zero_grad), and vector norm
implements the same RMS values while defining a finite subgradient at zero.
The redundant upstream mask forward whose result is immediately overwritten is
omitted. These are not claims of bitwise equivalence of the entire training run.

Attacker access is white-box joint training, including victim parameter gradients;
it is not claimed to be restricted to data-only poisoning. This adaptation must
be labelled separately in comparisons, including this access/compute difference.
