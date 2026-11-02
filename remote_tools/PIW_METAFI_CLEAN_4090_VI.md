# Clean MetaFi++ / PiW3D trên máy Vân RTX 4090

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, run preparation.
- Origin Date: 2026-11-03, Asia/Bangkok.
- Verification Status: **386 CPU/mock/regression tests PASS**, exit0, 188.31s
  ngày2026-11-03; gồm sidecar4090, Clean3090 và completed3090 evaluator.
  Chưa có full PiW/CUDA run trên máy Vân từ workspace Admin. Không gọi local
  CPU/mock tests là GPU PASS. XML local:
  `E:/wifi/deliverables/clean4090_regression_20261103_b.xml`.
- Version: piw_metafi_clean_4090_diagnostic_v1.

Người dùng yêu cầu thử 4090 sau khi Clean 3090 env cũ và env khớp Vast đều
~200.126 mm, trong khi reference Vast lịch sử ~136.164 mm. Hai phép thử tách rõ:

1. Evaluate **chính checkpoint Clean 3090 env-matched final199**, cùng 2586
   ordered TEST, trên 4090. Chỉ inference, không sửa trọng số/BN, optimizer,
   RNG hoặc execution metadata, không dùng checkpoint này để resume training.
2. Train Clean 4090 **mới từ scratch** (vẫn pretrained ResNet34 V1 như recipe),
   seed42, AdamW lr=.01/wd=.01/eps1e-8, B32, FP32, workers0, 200epoch/final199.
   Không A1/baseline, không LR scheduler/AMP/early stop hoặc chọn best TEST.

Mỗi pha GPU chỉ một worker, chặn compute lạ, không kill/retry/bỏ guard.
Foreground: giữ SSH mở. Không có tmux/Task Scheduler/WMI hoặc tự sống qua SSH.
Checkpoint train mới lưu mỗi epoch; chỉ resume chính profile4090 này khi source,
plan/runtime/GPU/optimizer/RNG bindings khớp, không import checkpoint3090.

## Nguồn và nơi lưu

- Source3090: `backdooranalog/runs/piw_metafi_clean_vastenv_s42_3090_v1`.
- Export3090: `backdooranalog/exports/piw_metafi_clean_3090_reference_v1`.
- Reference trên Vân: `backdooranalog/imports/piw_metafi_clean_3090_reference_v1`.
- Env Vân mới: `backdooranalog/envs/piw-metafi-vast211-4090`.
- Output Vân mới: `backdooranalog/runs/piw_metafi_clean_vastenv_s42_4090_v1`.
- Replay riêng: `.../replay_3090/evaluation.json` + full paired predictions NPZ.
- Fresh Clean: `.../attacks/metafiplusplus/seed42/clean`.
- Bảng cuối: `.../comparison.md` và `comparison.json` có bốn dòng: Vast cũ,
  source3090, cùng checkpoint3090 eval trên4090, Clean4090 train mới final199.

Không sửa launcher3090, bản scientific source snapshot hoặc các run lịch sử.
Wrapper4090 chỉ thay profile/GPU guard/code binding/report orchestration;
native train/eval/model/data/optimizer source giữ nguyên hash đã đóng băng.
Gate4090 còn ghi Windows build và NVIDIA driver_version hiện tại; đổi driver
sau Prepare bị chặn bởi binding, không tự coi driver3090/Vast lịch sử giống nhau.

## 1. Lấy code trong checkout MỚI trên mỗi máy

Không pull vào checkout mà job đang chạy. Dùng SHA được bàn giao trong chat;
thay `PUBLISHED_COMMIT_SHA` bên dưới bằng SHA đó, không dùng literal placeholder.

```powershell
$ErrorActionPreference = 'Stop'
$Rev = 'PUBLISHED_COMMIT_SHA'
$Code = "$env:USERPROFILE\code_metafi_clean4090_diag_20261103"
if (Test-Path -LiteralPath $Code) { throw 'Checkout da ton tai; khong ghi de.' }
git clone --single-branch https://github.com/toan31072006/backdoor-analog.git $Code
if ($LASTEXITCODE -ne 0) { throw 'Clone loi; dung.' }
git -C $Code checkout --detach $Rev
if ($LASTEXITCODE -ne 0) { throw 'Pin commit loi; dung.' }
$S = "$Code\remote_tools\Start-PiWMetaFiClean4090.ps1"
```

Máy3090 từng thiếu Git: có thể tải source ZIP từ codeload tới một tên mới,
không có checkpoint/data trong ZIP này và không giải nén đè thư mục cũ:

```powershell
$ErrorActionPreference = 'Stop'
$Rev = 'PUBLISHED_COMMIT_SHA'
$Zip = "$env:USERPROFILE\Downloads\clean4090_diag_$Rev.zip"
$Dir = "$env:USERPROFILE\code_metafi_clean4090_zip_20261103"
if ((Test-Path -LiteralPath $Zip) -or (Test-Path -LiteralPath $Dir)) {
    throw 'Duong dan da ton tai; dung, khong ghi de.'
}
Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/toan31072006/backdoor-analog/zip/$Rev" -OutFile $Zip
Expand-Archive -LiteralPath $Zip -DestinationPath $Dir
$S = "$Dir\backdoor-analog-$Rev\remote_tools\Start-PiWMetaFiClean4090.ps1"
```

## 2. Trên máy3090: xuất reference đã train xong, không dùng GPU

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Export3090
```

Export chỉ chấp nhận đúng run Clean env-matched3090 đã COMPLETE, giữ nguyên
checkpoint bytes (~318 MB, có optimizer/RNG), plan, gate, receipt, evaluation và
predictions. Không export A1, không nạp pickle/torch, không train, không ghi lại
source artifacts. In `MANIFEST_SHA256`; ghi lại giá trị này để kiểm sau chuyển.
Nếu đích đã tồn tại, chỉ xác minh bản export hoàn chỉnh; bản dở hoặc không khớp
sẽ bị chặn. Không xóa/ghi đè/tự retry. Cần dung lượng cho thêm một bản checkpoint.

Chuyển nguyên thư mục reference sang Vân (không cần ZIP). Ví dụ trên3090,
sau khi Vân đã có thư mục `C:\Users\Van\backdooranalog\imports`:

```powershell
scp -r "$env:USERPROFILE\backdooranalog\exports\piw_metafi_clean_3090_reference_v1" 'van@100.100.233.102:backdooranalog/imports/'
```

Đây là thao tác chuyển file trực tiếp giữa hai máy của người dùng; có thể hỏi
mật khẩu SSH tương tác, không gửi mật khẩu trong chat. Kiểm đích chưa có bản
cũ trước khi copy, không tự chép đè. Nếu không kết nối được, tải thư mục về máy
Admin rồi upload sang Vân. Không upload checkpoint/dataset lên GitHub.
Chỉ dùng pickle checkpoint do chính người dùng tạo và đã kiểm manifest, không
dùng file tải từ người lạ. Manifest là kiểm tra tính toàn vẹn, không chữ ký tác giả.

## 3. Trên Vân: env riêng và Prepare

Base Python phải đúng **3.12.15**, giống run3090. Mặc định tìm
`C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`; kiểm trước:

```powershell
& 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe' -c "import sys; print(sys.version); print(sys.executable)"
```

Nếu khác patch version, dừng và gửi output; không thay env đang train hoặc
tự cài phiên bản khác. Có thể chỉ định Python3.12.15 đã có bằng `-BasePython`;
dùng lại cùng đối số ở các lần Setup sau. Không kích hoạt đè base env.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Setup
if ($LASTEXITCODE -ne 0) { throw 'Setup loi; dung.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; dung.' }
```

Setup dùng lại installer cách ly đã test: Python3.12.15, torch2.11.0+cu128,
torchvision0.26.0+cu128, numpy2.5.4, scipy1.17.1, PyYAML6.0.2. Giữ chính sách
2 OMP/MKL threads, MKL SEQUENTIAL; TF32 tắt, FP32/highest theo native worker.
Không sửa driver hoặc Python/env cũ; không nhận symlink/junction/path escape.

Prepare kiểm full input content/identity/order và source, rồi CUDA B32 smoke
thật (1 update trong RAM, không lưu checkpoint); chỉ xác nhận chạy được, không
hứa final199 sẽ giống Vast. Full content audit đầu tiên của profile4090 được
lưu rồi reuse với fingerprint/mtime checks ở các pha sau; không rehash lại mỗi
epoch. Phải có dữ liệu full TRAIN27876/TEST2586 và action data_bend.npy đúng
nguồn, không tự download data khác hoặc đổi split/subset.

## 4. Replay cùng checkpoint3090, chưa train Clean mới

Sau khi copy xong, kiểm manifest file trên Vân khớp giá trị máy3090 đã in:

```powershell
Get-FileHash -LiteralPath 'C:\Users\Van\backdooranalog\imports\piw_metafi_clean_3090_reference_v1\reference_manifest.json' -Algorithm SHA256
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Replay3090
```

Replay so full ordered IDs và GT phải exact, rồi ghi MPJPE/PA trên4090, delta
metric, mean paired prediction difference và max coordinate difference (mm).
So model/BN state trước/sau inference phải exact. Không thay metadata GPU của
checkpoint để vượt native training guard và không restore optimizer/RNG.

`COMPLETE` chỉ nghĩa inference hoàn tất; không tự đặt ngưỡng similarity từ
TEST hoặc coi đó là proof tái lập training. Kiểm metadata nguồn từ saved arrays
có tolerance0.001mm chỉ để phát hiện artifact sai, không phải ngưỡng so GPU.

```powershell
$R = 'C:\Users\Van\backdooranalog\runs\piw_metafi_clean_vastenv_s42_4090_v1'
[IO.File]::ReadAllText("$R\replay_3090\evaluation.json")
```

## 5. Full Clean mới: Launch riêng sau khi xem replay/smoke

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
if ($LASTEXITCODE -ne 0) { throw 'Gate loi; dung.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Launch yêu cầu đúng receipt replay đã gắn với reference/source/GPU/plan hiện
tại. Độ cao/thấp metric replay không được dùng để đổi recipe hoặc chọn model.
Chỉ train Clean, 1job/GPU. Chặn GPU compute khác, không tự dừng chúng.
Nếu SSH bị ngắt hoặc job lỗi, giữ log/checkpoint; không tự retry. Người dùng có
thể gọi Launch lại để resume chính run4090 nếu bindings không đổi. Khi đã có
evaluation hợp lệ, chỉ in lại bảng của run4090, không train lại từ đầu.

Xem trạng thái trong SSH khác, không launch trùng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
Get-ChildItem -LiteralPath "$R\logs" -Filter 'launch.*.log' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Tail 15 -Encoding UTF8 }
```

Không coi GPU0% hoặc PID còn sống là bằng chứng train xong. Dựa vào epoch/log,
TRAIN_COMPLETE, EVAL_COMPLETE, final199 artifact binding và comparison report.
Không có early stopping tự động hoặc kill ở mốc30 phút: full train200epoch
có thể mất nhiều giờ; thời lượng thực tế theo epoch trên máy Vân.

## Giới hạn diễn giải

Đây là diagnostic một seed về nền tảng, không chứng minh riêng3090/GPU bị lỗi.
Windows/driver/cuDNN/backend có thể vẫn khác Vast/Linux và khác giữa hai máy.
Inference gần nhau không chứng minh training trajectories sẽ giống nhau;
smoke loss gần nhau cũng không chứng minh final199 giống nhau. Không cập nhật
bảng paper hoặc file kết quả lịch sử chỉ vì một run mới cho số đẹp hơn.
