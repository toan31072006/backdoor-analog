# MetaFi++ / PiW3D / bốn baseline / seed42 / 4090

## Material Passport

- 2026-11-04, academic-research-suite / experiment-agent.
- Selected-method T1 extension, một seed; không phải MM-Fi hoặc bảng đa seed.
- Yêu cầu: chỉ FIBA-CSI, WaNet-CSI, FTrojan-CSI, BppAttack-CSI; không thêm
  TSBA, Smooth LF, BadNets hay job Clean/A1.
- Code scientific frozen Vast không thay đổi. Clean và A1 lịch sử chỉ đọc.

## Recipe và scope

Chạy tuần tự **FIBA -> WaNet -> FTrojan -> BPP**, một foreground worker GPU0.
Mỗi victim khởi tạo ImageNet ResNet34 V1 + MetaFi init như Clean/A1, AdamW
lr0.01/wd0.01,200 epoch/final199,B32,FP32,workers0,drop_lastTrue,seed42.
Không scheduler/AMP/chọn best TEST, không warm-start từ Clean, không migrate
optimizer/checkpoint 3090. Resume chỉ từ checkpoint của chính queue mới.

Full PiW3D TRAIN27876/TEST2586, rho0.1/2787 poison, cùng body-hip-axis v2,
target joints11/13 và GT PCK scale6/4, sáu dose0,.2,.4,.6,.8,1. Các native
cover ratios giữ nguyên: FIBA/BPP0.1,WaNet0.2,FTrojan0; không equal-noise claim.
Giữ canonical TRAIN donor IDs của FIBA/BPP và pattern FTrojan Linux frozen.
Native three-group amplitude operators, phase unchanged; không thiết kế trigger mới.

Env hiện có `envs/piw-metafi-vast211-4090`: Python3.12.15,torch2.11.0+cu128,
torchvision0.26.0+cu128,numpy2.5.4,scipy1.17.1. Launcher không cài lại env/data.
Tái sử dụng full audit Clean, kiểm tra sources/runtime/reference hashes,
official lists/action, all file size/mtime, ordered IDs và actual poison/cover keys.
Không hash lại mọi nội dung CSI/pose; stat checking không là attestation chống sửa cố ý.

Prepare chạy bốn native real-CUDA B32 smokes nối tiếp. FTrojan mix4poison/28clean;
các method có cover dùng4poison/1cover/27clean. Không lưu training checkpoint.
Launch gọi riêng, hoàn tất cả bốn final199 trước khi TEST, rồi evaluate từng
method với cùng Clean4090 external reference, đủ six-dose/no-trigger controls.
Checkpoint lưu mỗi epoch với optimizer/RNG và ba operator states. Lỗi dừng queue,
không tự retry/kill, không launch method sau. Giữ SSH mở trên native Windows.

## Lệnh trên Vân

Từ checkout mới, không pull/sửa checkout lịch sử:

```powershell
$S = 'C:\Users\Van\code_piw_metafi_four4090_v1\remote_tools\Start-PiWMetaFiFourAttacks4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

Prepare đạt thì gọi riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Resume dùng cùng Launch, không Prepare/git pull/reinstall. Không xóa lock hoặc
gate để ép chạy. Nếu còn job GPU khác thì dừng, không tự tắt process.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Output `runs/piw_metafi_four_attacks_vastenv_s42_4090_v1`:
`logs/`, `smokes/`, `prelaunch_gate.json`,
`attacks/metafiplusplus/seed42/<method>/checkpoint.pt`, `evaluation.json`,
`predictions.npz`; cuối cùng `RESULT.md` và `results.json`.
Report dùng Clean4090; nếu A1 đã hoàn tất ở run chuẩn và các hash khớp, thêm
hàng A1 chỉ đọc, không train/evaluate lại nó. Không dùng số HPE-Li/MM-Fi được
dán trong yêu cầu làm metric MetaFi/PiW3D. Main T/PCK tại dose1, DCE mean năm dose dương.

Local CPU/mock tests không thay cho Prepare CUDA/full-data trên máy Vân.

## Kiểm chứng local

2026-11-04: 204 tests launcher mới/A1/controller đạt; thêm 7 tests bảo toàn
A1 và tổng hợp report đạt (211 kiểm tra phân biệt). Frozen native function
identity, bốn operator/phase/cover-state replay, FTrojan Linux key, external
Clean six-dose metric parity trên dữ liệu synthetic, serial scheduling,
resume/fail-stop, recipe guards và PowerShell parser đều đạt.
Chưa chạy full dataset/CUDA smoke/training trên máy Vân từ workspace Admin.
