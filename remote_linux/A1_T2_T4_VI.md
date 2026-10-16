# Bảng 2 và bảng 4 — MM-Fi seed 42

Tiếp tục recipe A1 đã chốt: HPELi, MM-Fi protocol1-s1, split seed 0,
133056 TRAIN / 33264 TEST, seed 42, SGD lr=.001/momentum=.9, batch 32,
50 epochs, paired U(.2,1), pivot 1 → joints [2,3], góc Z 40 độ,
`md_multicarrier` native. Không thay đổi các baseline đã train.

## T2: Dose × Poison-rate

| rho | T(d=0) | T(.2) | T(.4) | T(.6) | T(.8) | T(1) | Clean MPJPE | mean-positive DCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .1 | reuse A1 | reuse | reuse | reuse | reuse | reuse | reuse | reuse |
| .2 | train victim mới 50 epochs rồi evaluate đủ 6 dose | | | | | | | |
| .4 | train victim mới 50 epochs rồi evaluate đủ 6 dose | | | | | | | |

Mỗi ô T là T-MPJPE trên các khớp mục tiêu, đơn vị mm. d=0 là identity control;
Clean MPJPE chấm cả skeleton. DCE lấy trung bình 5 dose dương, không dùng d=0
để làm nhỏ giá trị trung bình. Full và coupling control rho=.1 từ factorial
được giữ trong các đường T-MPJPE/DCE; không train lại hai model đó.

T2 chỉ có **hai victim mới**. Các run peak-bound/rho=.4 lịch sử không được nhập
vào bảng này. New runs lưu checkpoint và có thể resume đúng fingerprint.

Output: `runs/mmfi_a1_dose_poison_s42_v1/`, gồm bảng T2 CSV/JSON/Markdown,
six-dose data và hai plot T-MPJPE/DCE.

## T4: Stealthiness

Reuse 7 targeted checkpoints T1: BadNets, WaNet, FTrojan, FIBA, SIG, BppAttack,
Proposed A1, tất cả rho=.1. Không train lại victim.

- Đo paired distortion trên **toàn bộ 33264 TEST frames** tại d=1. Relative L2
  pool signal/error energies trong domain input thực tế của victim. Raw processed
  CSI, RMSE, Linf và SNR là diagnostics.
- Input detector: Isolation Forest, 200 trees, max_samples=256, seed42,
  12 statistics/link, không fit bằng trigger labels.
- Feature detector: frozen HPELi pooled-SK2 output 128D; StandardScaler,
  PCA tối đa 64D và regularized LedoitWolf Mahalanobis fit bằng clean TRAIN.
- Trusted clean TRAIN chia **theo E/S/A sequence**, 80% nhóm fit và 20% nhóm
  calibration, rồi lấy tối đa 4096/2048 frames không lặp. Các bank dùng chung
  giữa methods và lưu đầy đủ IDs. Đây không phải một held-out-subject protocol.
- Threshold là higher quantile .95 trên clean calibration; cảnh báo khi score
  **lớn hơn** threshold. Báo raw AUROC, TPR và **FPR thật trên TEST**. Không đổi
  chiều score hoặc chọn threshold theo TEST.

Bảng chính: Relative L2, AUROC-input và AUROC-feature. Panel/CSV đi kèm có
TPR@calibrated-5%-FPR và actual TEST FPR cho từng detector. AUROC gần .5 chỉ cho
thấy detector này phân biệt kém; kết quả digital CSI không chứng minh RF/OTA.

Output: `runs/mmfi_a1_stealthiness_s42_v1/`, gồm bảng, bank IDs, per-frame
clean/trigger detector scores + distortion energies, fitted detector artifacts,
config/checkpoint/source hashes. Không có kết quả mới trước khi chạy trên MICA.

## Pull và chạy ba GPU trên MICA

Chọn GPU vật lý đang có thể dùng. Ví dụ sau dùng 0 và 2 cho hai victim T2,
GPU 3 cho inference T4. Các logical IDs tự được map trong mỗi job.

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
"$HOME/backdooranalog/envs/dose-backdoor/bin/python" -m pip install -r remote_linux/requirements_t4.txt
nvidia-smi
tmux new-session -d -s mmfi-t2t4-s42 \
  -c "$HOME/backdooranalog/code" \
  'DOSE_TRAIN_GPUS=0,2 DOSE_EVAL_GPU=3 bash remote_linux/22_run_a1_t2_t4.sh'
```

Không chạy thêm seed 43/44 ở lượt này. T4 có thể hoàn tất trước T2.

Chạy riêng từng bảng nếu cần:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
bash remote_linux/20_run_a1_dose_poison.sh --devices cuda:0 cuda:1 --num-workers 4

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
bash remote_linux/21_run_a1_stealthiness.sh --device cuda:0
```

Thêm `--dry-run` để kiểm tra plan/metadata; không khởi chạy data/GPU/training.

## Theo dõi

```bash
tail -n 25 "$HOME/backdooranalog/runs/mmfi_a1_dose_poison_s42_v1.startup.log"
tail -n 25 "$HOME/backdooranalog/runs/mmfi_a1_stealthiness_s42_v1.startup.log"
tail -n 8 "$HOME/backdooranalog/runs/mmfi_a1_dose_poison_s42_v1"/*/console.log
```

## Đóng ZIP khi cả hai bảng hoàn tất

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_a1_t2_t4_s42_v1_results.zip \
  mmfi_a1_dose_poison_s42_v1 mmfi_a1_stealthiness_s42_v1 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png' '*.npz' '*.joblib'
unzip -tq mmfi_a1_t2_t4_s42_v1_results.zip
ls -lh mmfi_a1_t2_t4_s42_v1_results.zip
```
