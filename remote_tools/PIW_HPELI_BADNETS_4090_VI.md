# BadNets-CSI / HPE-Li / Person-in-WiFi-3D / seed42 / Vân4090

## Phạm vi và recipe

Chỉ thêm một victim BadNets cho PiW3D, không phải MM-Fi. Đây là phần bổ sung
selected-method một seed; không gọi là full bảng paper hay mean/SD nhiều seed.
Không train lại Clean/A1 và không sửa source, receipt hay checkpoint lịch sử.

- Reference read-only: `runs/piw_hpeli_body_axis_s42_lr001_v3`, hoàn tất Clean+A1
  final199, đầy đủ manifest/checkpoint/NPZ/poison và hash. Source/runtime của
  reference phải qua verifier v3 nguyên bản; mismatch thì dừng, không vá hash.
- HPE-Li scratch, AdamW lr0.001/wd0.01, B32, workers0, FP32, fixed LR,
  drop_last, 200epoch/final199. Không dùng SGD/MM-Fi hoặc MetaFi/PiW lr0.01.
- TRAIN27876/TEST2586, official one-person identities, normalization nguyên bản.
  Cùng floor(0.1*TRAIN)=2787 poison indices và paired U(.2,1) doses của A1 v3.
- Target `piw_body_hip_axis_v2`: pivot7, joints[11,13], axis
  normalize(clean_GT[5]-clean_GT[7]), 40deg*d. Không dùng MM-Fi [2,3] hoặc Z-axis
  PiW v1. Không chọn label/target bằng điểm TEST.
- Native BadNets white3x3 patch trên từng 3x30x20 physical amplitude group,
  tất cả3antennas, góc cuối [27:30,17:20], opacity=dose. Phase90rows giữ nguyên,
  không normalize lại. Dùng upstream masked replacement trong repo, không patch
  ảnh/đổi antenna thành màu. Không cover. `attack_eps=.185` chỉ là interface
  chung, KHÔNG cap patch; không claim equal-L2/Linf với A1 hay RF/over-air.
- Sáu dose0,.2,.4,.6,.8,1. T-MPJPE/Target-PCK chỉ joints[11,13] tại d1;
  PCK0.2 scale clean_GT(6,4), không phải ASR. Clean MPJPE/PA14joints;
  DCE trung bình5dose dương. Lưu paired prediction/GT/targets/names, clean-model
  +trigger controls và same-victim no-trigger controls trong NPZ/JSON.

## Giữ job MM-Fi đang chạy

Ảnh người dùng gửi có BadNets/WiFlow MM-Fi PID24476. Không kill, pause hoặc
cho BadNets/PiW chạy chồng nó. Check GPU-util thấp không chứng minh GPU rảnh.
CPU `Preflight` chạy được trong lúc đợi, nhưng còn chia sẻ CPU/đĩa.
`Prepare`/`Launch` chặn compute/pending experiments khác, kể cả job MetaFi mà
queue MM-Fi có thể mở sau WiFlow. Không có bypass/concurrency flag.

Lấy code vào **checkout mới** `C:\Users\Van\code_piw_badnets`; tuyệt đối không
pull/thêm file vào `code_badnets_mmfi`, `code_motion_priors` hoặc repo đang train.
Bind source mới bao gồm scientific dependencies và launcher. Sau Prepare không
pull/sửa checkout mới nếu cần resume với receipt đã khóa.

```powershell
$S = 'C:\Users\Van\code_piw_badnets\remote_tools\Start-PiWHPELiBadNets4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Preflight
```

Preflight chỉ hash/kiểm tra CPU, không full train hoặc CUDA. Nếu reference nguồn
cũ khác snapshot mới, gửi nguyên lỗi; không đổi manifest hay tạo reference giả.
Khi toàn bộ queue MM-Fi đã xong và không còn compute/pending experiment khác:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua train.' }
```

Prepare chạy full audit rồi một disposable HPE-Li B32 CUDA smoke trên TRAIN thật,
4poison+28clean, 2optimizersteps. Nó dừng ở PREPARED_ONLY, không khởi chạy train.
Failed/RUNNING gate được giữ nguyên, không automatic retry/overwrite failed gate.
Sau xem receipt PASS, launch riêng trong SSH giữ mở:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Một blocking worker train rồi evaluate final199. Checkpoint atomic từng epoch,
optimizer/RNG/poison/operator/execution binding. Launch lại cùng snapshot/output
chỉ resume checkpoint hợp lệ; log train tồn tại mà checkpoint mất thì dừng.
Ctrl+C sẽ drain worker sở hữu hiện tại, không kill; không đóng SSH trong lúc
drain. Không bảo đảm sống sau mất SSH/reboot; không tạo Task Scheduler/WMI/tmux.
Không có OOM batch fallback, AMP, chọn best TEST, automatic job retry hay kill.

## Xem tiến độ và kết quả

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Status chỉ đọc log/file markers, không chạm CUDA và không tự chứng minh live PID.
Output mặc định `runs/piw_hpeli_badnets_s42_4090_v1`:

- `badnets.plan.json`, sibling `.preflight.json` và `.gpu_smoke.json`.
- `hpeli/seed42/badnets/train.*.log`, `evaluate.*.log`, `checkpoint.pt`.
- `hpeli/seed42/badnets/evaluation.json`, `predictions.npz`.
- `badnets_results.json` chỉ sau final199 + evaluation artifact replay thành công.

Verification: local CPU/synthetic/mocked checks, không phải actual full PiW/CUDA
hoặc efficacy mới. Origin Skill: academic-research-suite / experiment-agent;
scope triển khai được user yêu cầu, không thay recipe để ưu tiên kết quả đẹp.
