# A1 / MetaFi++ / Person-in-WiFi-3D / seed42 / RTX4090

Chỉ train A1 mới; dùng Clean đã hoàn tất tại
`backdooranalog/runs/piw_metafi_clean_vastenv_s42_4090_v1` làm đối chứng chỉ đọc.
Không phải MM-Fi P3S3. Không sửa hay retrain Clean, không lấy Clean để warm-start A1.

- Giữ nguyên trainer/model/trigger/target/loss từ snapshot Vast: ResNet34 ImageNet V1
  + MetaFi init, AdamW lr0.01/wd0.01, 200 epoch/final199, B32, FP32, workers0,
  drop_last, seed42. Không scheduler, AMP, chọn best TEST hay đổi batch.
- A1 PiW body-hip-axis v2, rho0.1 (2787/27876 TRAIN), paired dose U(.2,1),
  amplitude gain/clip gốc, phase không đổi. Không thêm baseline khác.
- Dùng đúng env Clean hiện có `envs/piw-metafi-vast211-4090`:
  Python3.12.15, torch2.11.0+cu128, torchvision0.26.0+cu128,
  numpy2.5.4, scipy1.17.1. Không cài/chỉnh env ở launcher này.
- Prepare tái sử dụng full audit của Clean: kiểm tra source/runtime, checkpoint,
  evaluation, official lists, trigger/action, thứ tự identity, poison manifest và
  size/mtime tất cả file. Không hash lại toàn bộ nội dung CSI/pose. File thay đổi
  sẽ dừng; stat checking không phải attestation chống can thiệp.
- CUDA smoke B32 thật dùng 4 poison +28 clean; không lưu checkpoint, không train
  đủ. Launch phải gọi riêng. Không có bảo đảm tái lập số Vast hay mức efficacy.
- Một worker GPU, foreground, giữ SSH mở. Không tự kill/retry job, không chạy
  ghép job GPU khác. Resume bằng cùng lệnh Launch, chỉ từ checkpoint của run A1
  này (có optimizer/RNG/execution); không migrate checkpoint 3090/Vast.

Trên PowerShell máy Vân, sau khi lấy **checkout mới** chứa launcher:

```powershell
$S = 'C:\Users\Van\code_piw_metafi_a1_4090_v1\remote_tools\Start-PiWMetaFiA14090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

Prepare đạt thì gọi riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Terminal khác, kiểm tra log/kết quả (chỉ đọc):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Output mới `runs/piw_metafi_a1_vastenv_s42_4090_v1`:
`logs/`, `a1_smoke.json`, `prelaunch_gate.json`, checkpoint mỗi epoch tại
`attacks/metafiplusplus/seed42/a1/checkpoint.pt`, sau final199 có
`evaluation.json`, `predictions.npz` và `RESULT.md` hai hàng Clean/A1.
Evaluation full TEST2586 ở sáu dose, cùng metric native và các control
same-victim no-trigger/Clean+trigger. T-MPJPE/Target-PCK ở dose1; DCE mean năm
dose dương. Một seed, không phải bảng mean±std đa seed.

Kiểm chứng local 2026-11-03: **348 tests passed** gồm launcher A1 mới,
hai launcher Clean 3090/4090 và controller frozen. Có kiểm tra identity các
hàm scientific native, metric synthetic và ghép evaluation external Clean;
PowerShell parser đạt. Không chạy train/evaluate trên GPU local hoặc remote.
Các CPU/mock tests này không thay cho Prepare CUDA/full-data trên máy Vân.
