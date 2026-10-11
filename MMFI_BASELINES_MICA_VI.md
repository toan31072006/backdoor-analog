# Sáu baseline MM-Fi trên MICA — DT-Pose/HPELi

Entry point: `remote_linux/14_run_mmfi_baselines.sh` → `run_baselines.py`.
Chỉ chạy MM-Fi: BadNets-CSI, Blended-CSI, WaNet-CSI, FTrojan-CSI, FIBA-CSI,
INFOCOM2025/POR. Clean teacher là phụ thuộc tự động của POR, không phải attack.
Không lên lịch Proposed/CCAI/PiW3D/WiPose. Không sửa operator baseline hoặc trainer.

## Nguồn config

[DT-Pose MM-Fi YAML](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/config/mmfi/pose_config.yaml)
được lưu nguyên nội dung Git/LF trong `original_configs/dtpose_mmfi_original.yaml`.
SHA256: `237d6097c6106052abf0bd5b79cc2e26e3f7f05fa5062ca53d9bd846785e0997`.

Dùng nhánh [HPELi scratch](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L106):
50 epochs, seed42, batch32, SGD lr0.001/momentum0.9/weight_decay0,
MPJPE loss, không scheduler. P1/S1, train ratio0.8, splitseed0, 17khớp,
CSI3×114×10. Không dùng pretrained Transformer mặc định của YAML DT-Pose.

Đây là config DT-Pose/HPELi + attack trainer đã adapted, không phải tái lập
nguyên toàn bộ DT-Pose. Giữ loader297frame/sequence, workers4/drop_lastFalse,
modelseed42; DT-Pose nguồn dùng296TRAINframe, workers8/drop_lastTrue và
resetTorchseed0 trong loader. Không thay các chi tiết này trong lần deploy.

Config chạy: ROOT `configs/mmfi/*.yaml`, không chọn nhầm `ATKBackd/configs/` cũ.
Năm targeted baseline giữ nguyên profile đã chạy: rho0.1/uniform, cùng paired
dose U(0.2,1), theta40°, pivot1/target[2,3], sáu dose đánh giá0,0.2,...,1.
Giữ native method knobs và existing shared-peak-only ceiling/reference eps0.185;
không thêm L2 cap. Peak ceiling chung không đồng nghĩa cùng realized L2.

POR giữ profile riêng: rho0.4/fixed training dose1/amplitude0.185;
teacher50 → encoder Adam0.001×50 → head SGD0.001×50. Có supervised teacher
và downstream TRAIN CSI không nhãn, không phải data-free reproduction.
Không xếp hạng T-MPJPE của POR cùng năm targeted baseline.
WaNet/FIBA workers0 để giữ RNG cover/resume, các method khác mặc định4.
Provenance máy đọc được: `MMFI_BASELINES_SOURCE.json`.

## Pull và chạy trên MICA

Data/env hiện có:

```text
~/backdooranalog/datasets/Compress/
~/backdooranalog/actions/data_bend.npy
~/backdooranalog/envs/dose-backdoor/bin/python
```

Trong terminal MICA:

```bash
cd ~/backdooranalog/code
git pull --ff-only
git log -1 --oneline

# Chỉ lập plan; không mở data/GPU hoặc train.
bash remote_linux/14_run_mmfi_baselines.sh --dry-run

# Chọn lại GPU theo tình trạng thực tế. Ví dụ physical0,1,3 → logical0,1,2.
nvidia-smi
tmux new-session -d -s mmfi-six-dtpose -c "$PWD" \
  'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3 bash remote_linux/14_run_mmfi_baselines.sh > "$HOME/backdooranalog/runs/mmfi_baselines_dtpose_s42_v1.startup.log" 2>&1'
```

Log startup nằm BÊN CẠNH output, không tạo file trong output mới trước manifest.
Gói không tự dừng job khác. Chỉ dùng physical0,1,3 nếu được phép và đang phù hợp.
Nếu env thiếu deps, cài `remote_linux/requirements-linux.txt` bằng Python trong
env hiện có; không cài lại PyTorch/CUDA khi đang hoạt động.

```bash
tail -f ~/backdooranalog/runs/mmfi_baselines_dtpose_s42_v1.startup.log
tail -n 12 ~/backdooranalog/runs/mmfi_baselines_dtpose_s42_v1/*/console.log
```

Chạy lại cùng command/outdir để resume checkpoint/cache hợp lệ. Thay config/code
hoặc chọn method khác phải dùng DOSE_OUT mới. Không import manifest từ ZIP/Windows
hoặc cache của thí nghiệm khác: source hashes có thể khác khi clone Git trên Linux.

Ví dụ chỉ Blended, một GPU:

```bash
DOSE_OUT="$HOME/backdooranalog/runs/mmfi_dtpose_blended_only_s42_v1" \
CUDA_VISIBLE_DEVICES=3 bash remote_linux/14_run_mmfi_baselines.sh \
  --methods blended --devices cuda:0
```

DOSE_OUT/DOSE_DATA_HOME/DOSE_PYTHON đặt đường dẫn khác. Nếu đặt biến trước tmux,
phải export hoặc gán trực tiếp trong command con, không chỉ đặt biến shell cục bộ.

## Kết quả

Output mặc định `~/backdooranalog/runs/mmfi_baselines_dtpose_s42_v1/`:

- `baseline_metrics.md/csv`: năm targeted baseline, MPJPE, PA-MPJPE, T-MPJPE d=1,
  PCK0.5/0.4/0.3/0.2/0.1 (%), không loại baseline tốt hơn Proposed.
- `dose_response.csv`: sáu dose cho năm targeted baseline.
- `por_diagnostics.md/csv`, `por_dose_response.csv`: POR riêng.
- `clean_teacher_metrics.md/csv`: teacher hỗ trợ.
- `baseline_report.json`, `baselines.resolved.json`: complete status/plan/hash.
- Mỗi cell có `console.log`, `checkpoint.pt`, `eval_cache.json`.

Chỉ xuất báo cáo cuối khi tất cả cell được yêu cầu hoàn tất/cache khớp.
Full TRAIN133056/TEST33264, không cap/subset15epoch. Single seed, chưa có
thống kê ưu thế; digital distortion không chứng minh RF stealthiness/OTA.

BadNets/Blended là source-pinned BackdoorBench third-party, không phải code tác giả
paper. WaNet/frequency/POR là CSI/HPE adaptations đã có. Retained source/license
notices ở `ATKBackd/third_party/` và các module tương ứng vẫn áp dụng.
