# Chạy riêng A1 / HPE-Li / MM-Fi P3S3 seed42 trên 4090

Lấy code qua GitHub, không cần ZIP. Chỉ một scientific victim A1 được train;
không chạy Clean, S1/S2 hay attack khác. Scientific runner/operator/evaluator
giữ nguyên `ATKBackd/run_mmfi_p3_splits.py` của run3090 cũ, không port A1 sang
thuật toán khác. Đây là rerun một hàng, không phải bảng T6 hoàn chỉnh.

Recipe: SGD lr=.001, momentum=.9, wd0, B32, 50 epoch/final49, seed42,
rho=.1, A1 multi-carrier với paired U(.2,1), target pivot1->[2,3], Z40*d độ.
P3S3 cross-environment, splitseed0; không chọn lại hyperparameter theo TEST.
T-MPJPE/PCK@.2 tại dose1; DCE trung bình năm dose dương trong grid sáu dose;
Clean MPJPE/PA của chính A1 victim, không phải một clean-trained reference mới.

Trên PowerShell máy Vân:

```powershell
$Repo = 'C:\Users\Van\code'
git -C $Repo pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; dừng, không reset/xóa file local.' }
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3S3A1Only4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare lỗi; chưa train.' }
```

Prepare giữ preflight gốc, kiểm đầu vào cả ba setting P3 và chạy một physical
B32 CUDA smoke. Nó **không train** bất kỳ victim nào. Cần dữ liệu MM-Fi đầy đủ
ở `C:\Users\Van\backdooranalog\datasets\Compress` và prior thật ở
`actions\data_bend.npy`, SHA256
`2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e`.
Nếu GPU có job khác, helper dừng, không kill hay bypass guard.
Preflight gốc kiểm coverage/ordered identities của split và prior, không hash
toàn bộ nội dung CSI/pose. Không chỉnh dữ liệu giữa Prepare và Launch; đây
không phải xác nhận rằng mọi byte của dataset đã được audit lại.

## Giữ SIG và chạy A1 song song (user yêu cầu riêng)

Thêm `-AllowConcurrentSIG` ở cả Prepare/CheckOnly/Launch/Status. Đây là **hai
job trên một GPU: một A1 + một SIG**, không còn là GPU-exclusive. Không đổi
scientific recipe A1 hoặc SIG. Cờ này chỉ chấp nhận đúng một GPU process có
command line runner `run_mmfi_p3_s3_three_attacks_4090.py` trong checkout này,
`--method sig_source --mode cell --work train --device cuda:0`, đúng DataHome,
outdir `runs/mmfi_p3s3_three_attacks_s42_4090_v1` và jobs-per-gpu1. Input-Aware,
process Python khác hoặc nhiều SIG worker vẫn bị chặn. Không dừng SIG.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -AllowConcurrentSIG
```

Khi có PREPARED_ONLY:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -AllowConcurrentSIG
```

Nhánh có SIG dùng output riêng
`runs/mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1`, gate bind cả execution policy.
Không reuse gate GPU-exclusive cũ. Smoke B32 không tự sửa batch/AMP/recipe nếu
thiếu bộ nhớ. Nếu SIG kết thúc trước Launch thì A1 vẫn được chạy một mình.
Không bảo đảm hai job nhanh hơn hoặc cùng vừa VRAM; chưa có xác minh CUDA
concurrent trên máy Vân. Không pull/sửa source của SIG trong lúc nó đang train.

Khi có `PREPARED_ONLY`, chạy riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Job chạy ẩn, không cần giữ SSH; không tắt/sleep máy. Nó dùng entry cell đã
qua gate `p3_s3_a1`, không gọi controller train toàn bộ sáu cell. Output mới:
`C:\Users\Van\backdooranalog\runs\mmfi_p3s3_hpeli_a1_s42_4090_v1\p3_s3\a1`.
Giữ nguyên run3090 và các kết quả cũ. Train xong runner tự evaluate; artifact
gồm `checkpoint.pt`, `eval_cache.json`, `p3_evaluation.json`, `p3_per_sample.npz`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Nếu data home/env khác mặc định, dùng cùng `-DataHome`, `-PythonExe`, `-Outdir`
ở mọi lệnh. Helper ưu tiên envs/dose-backdoor/python.exe, sau đó
envs/mmfi-p3/Scripts/python.exe. Không cài/chỉnh môi trường có sẵn. Không pull
hay sửa code giữa Prepare và Launch. CheckOnly xác minh lại gate mà không train.

Lần chạy 4090 mới không bảo đảm bitwise cùng quỹ đạo 3090 hoặc đúng các scalar
334.49/6.43/85.30 trước đó. Không thay seed, chọn best TEST hay ghi đè kết quả cũ.
Nếu cần chạy lại sau crash, kiểm log/checkpoint và khả năng resume của engine
gốc trước; không tự xóa checkpoint/gate để bắt đầu lại.
