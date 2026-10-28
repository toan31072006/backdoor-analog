# HPE-Li/MM-Fi P3S3: BadNets, SIG và Input-Aware trên máy Vân

## Phạm vi

Theo yêu cầu mở rộng ngày 29/10/2026: train **ba victim mới**, seed 42,
**Protocol 3, Setting 3 (cross-environment)**. Đây là batch thăm dò/mở rộng
một seed, không phải bảng T1 nhiều seed đã hoàn thành. Không sửa hoặc train lại
Clean, A1, Smooth-LF hay TSBA đã chạy; không có Person-in-WiFi-3D/MetaFi++ hay
defense trong batch này.

Hàng chạy lần lượt trên `cuda:0`, tối đa **một job/GPU**:

1. `input_aware`: Input-Aware Dynamic Backdoor adapted.
2. `badnets`: BadNets-CSI adapted.
3. `sig_source`: SIG-CSI adapted.

Một slot là lựa chọn vận hành ban đầu, không phải kết luận tối ưu throughput.
Nếu máy Vân còn job PiW3D/MetaFi++ hoặc công việc của người khác, **không dừng
hay kill chúng**; xem `Inspect`, chờ GPU trống rồi mới `Prepare`/`Launch`.

## Hợp đồng thí nghiệm

- Victim HPE-Li giữ recipe P3 hiện có: SGD lr 0.001, momentum 0.9,
  weight decay 0, physical batch 32, 50 epoch, checkpoint cuối **49**;
  ordinary MPJPE, FP32, workers 0. Không AMP, auto-shrink batch, early stop
  hoặc chọn best checkpoint bằng TEST.
- Cùng official P3S3 và preprocessing MM-Fi, seed 42, poison rate 0.1,
  cùng poison identities và dose draws; trigger/payload paired U(0.2,1).
  Sáu dose đánh giá: 0, 0.2, 0.4, 0.6, 0.8, 1. Mỗi method train một victim,
  không train lại theo từng dose.
- Payload MM-Fi giữ pivot 1, target joints [2,3], Z40° × dose.
  Target-PCK@0.2 dùng scale từ clean GT joints (5,12), không dùng clean PCK
  thay Target-PCK và không gọi Target-PCK là pose-level ASR.
- Bảng `comparison.md` có T-MPJPE@1, Target-PCK@0.2@1, mean-positive DCE
  (chỉ 5 dose dương), Clean MPJPE, Clean PA-MPJPE. Giữ diagnostics
  no-trigger với cùng target d=1 và full six-dose artifacts để kiểm tra
  việc model chỉ tình cờ gần target; không kết luận hiệu quả chỉ từ một cột.
- TEST chỉ dùng đánh giá checkpoint final49, không dùng chọn hyperparameter,
  seed hay phương pháp. P3S3 đã có kết quả lịch sử; đây không phải blind test
  đầu tiên. Không hứa phương pháp mới sẽ mạnh hơn A1.

Các hàng là **CSI/pose/dose adaptations**, không phải tái lập classification
nguyên bản. BadNets là patch; SIG là pattern sin cố định. Input-Aware dùng
generator pattern + mask học từ đầu vào và cross-trigger training: cần quyền
white-box lúc huấn luyện, khác data-only poisoning. HPE victim vẫn 50 epoch,
nhưng Input-Aware có thêm **25 TRAIN-only mask-warmup passes**, optimizer
generator/mask Adam lr 0.01, betas (0.5,0.9), mask milestones 10/20,
gamma 0.1; các nhánh cross/cover được khai báo với ratio 0.1 và recipient
identities cố định. Đây là **compute/access khác**, không gọi equal-budget
chỉ vì cùng rho và lịch victim. Generator giữ ba downsampling stages,
pad CSI 114×10 thành 120×16 rồi crop về, ở miền model input [0,1].
Không thay A1 hoặc chỉnh các baseline sau khi nhìn TEST để làm đẹp bảng.
SIG ở đây dùng cùng pose-payload poisoning như các hàng khác; không giữ nguyên
clean-label/classification policy của SIG gốc.

## 1. Lấy code từ GitHub trên máy Vân, rồi Inspect

Repository: `https://github.com/toan31072006/backdoor-analog.git`.
**Không cần ZIP.** Chạy PowerShell trên máy Vân, không chạy trong Linux/Vast.
Chỉ code được clone/pull; dataset, env và output vẫn ở
`C:\Users\Van\backdooranalog` như trước.

Dùng checkout đã có ở **`C:\Users\Van\code`**, pull rồi Inspect bằng đoạn sau.
Nếu có tracked/untracked changes hoặc merge lỗi, dừng và giữ file; không
`reset`, `clean`, tự xóa file hoặc tự stash. Không pull checkout đang phục vụ
một job chạy dở; trường hợp đó dùng fallback clone riêng ở dưới:

```powershell
$Repo = Join-Path $env:USERPROFILE 'code'
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3S3ThreeAttacks4090.ps1'
& {
    $ErrorActionPreference = 'Stop'
    $Origin = git -C $Repo remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc repo/origin; dung.' }
    if ($Origin -notmatch '^https://github\.com/toan31072006/backdoor-analog(?:\.git)?/?$') {
        throw 'Origin khong dung repo du kien; chua pull.'
    }
    $Changes = @(git -C $Repo status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw 'Git status loi; dung.' }
    if ($Changes.Count -gt 0) {
        $Changes | ForEach-Object { Write-Host $_ }
        throw 'Checkout co thay doi/file untracked. Khong ghi de; dung fallback clone rieng ben duoi.'
    }
    $Branch = git -C $Repo branch --show-current
    if ($LASTEXITCODE -ne 0 -or $Branch -ne 'main') {
        throw 'Checkout khong o main. Khong tu switch branch; dung fallback clone rieng ben duoi.'
    }
    git -C $Repo pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Git pull loi; chua Inspect/Prepare/Launch. Giu nguyen file.' }
    git -C $Repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc commit; dung.' }
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
    if ($LASTEXITCODE -ne 0) { throw 'Inspect loi; chua Prepare/Launch.' }
}
```

**Fallback nếu checkout cũ bẩn, có file đụng hoặc đang phục vụ job khác:**
clone riêng, vẫn dùng dataset/env/output cũ ở `backdooranalog`. Đoạn này
dừng nếu thư mục đích đã tồn tại; không chép đè hay xóa checkout cũ:

```powershell
$Repo = Join-Path $env:USERPROFILE 'code-p3s3-20261029'
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3S3ThreeAttacks4090.ps1'
& {
    $ErrorActionPreference = 'Stop'
    if (Test-Path -LiteralPath $Repo) {
        throw 'Thu muc clone da ton tai; giu nguyen no va chon ten moi.'
    }
    git clone --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git $Repo
    if ($LASTEXITCODE -ne 0) { throw 'Clone loi; chua Inspect/Prepare/Launch.' }
    git -C $Repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc commit; dung.' }
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
    if ($LASTEXITCODE -ne 0) { throw 'Inspect loi; chua Prepare/Launch.' }
}
```

Giữ `$Repo`/`$S` của cách đã thành công cho các bước dưới. Nếu mở terminal
mới, đặt lại hai biến đó theo đúng checkout vừa dùng. Inspect chỉ đọc
env/data/GPU/dung lượng, không bắt đầu full training.

Launcher mặc định dùng data home `C:\Users\Van\backdooranalog`; tìm Python có
sẵn lần lượt ở `envs\dose-backdoor\python.exe` và
`envs\mmfi-p3\Scripts\python.exe`. Không tự cài/nâng package, tải dataset,
pull Git hoặc tạo môi trường. Muốn chọn env khác, thêm `-PythonExe` cùng giá
trị trong **tất cả** các lệnh tiếp theo. Python phải import đủ dependencies
và thấy CUDA; không fallback CPU. Dữ liệu MM-Fi và `actions\data_bend.npy`
phải có sẵn. Inspect không xác nhận đủ scientific input/content; Prepare
mới chạy full audit và real GPU smoke.

## 2. Prepare — chưa chạy full training

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua Launch.' }
```

Prepare kiểm split/content, source/runtime/config/prior và từng operator;
chỉ mở gate khi actual CUDA physical-batch-32 smoke thành công. Nó không
train ba victim đủ 50 epoch. Local CPU/synthetic tests trước khi publish không
thay smoke trên đúng RTX 4090 và dataset thật. Nếu lỗi/OOM, giữ log và dừng;
không xóa receipt, đổi batch hay tự retry để vượt kiểm tra.

## 3. Launch riêng sau Prepare thành công

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
if ($LASTEXITCODE -ne 0) { throw 'Gate loi; chua Launch.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
if ($LASTEXITCODE -ne 0) { throw 'Launch loi; xem controller log.' }
```

Launch tự kiểm gate lần nữa rồi mở controller ẩn, stdout/stderr tên duy nhất.
Controller giữ lock và chạy nối tiếp ba method. Không chạy `Launch` lần hai
khi controller còn sống. `STARTED` chỉ xác nhận process đã mở; cần log thay
đổi để xác nhận tiến độ. Windows/remote-session policy có thể ảnh hưởng
process khi logout; nối lại và kiểm tra Status, không mặc định rằng đã xong.

Thư mục output mới mặc định:
`C:\Users\Van\backdooranalog\runs\mmfi_p3s3_three_attacks_s42_4090_v1`.
Không tự sửa source/plan trong lúc chạy, không giết Python theo tên và không
xóa khóa/checkpoint khi gặp mismatch. Muốn resume sau sự cố, kiểm log/gate
trước rồi gọi lại cùng source, env, data và output; không tự retry thất bại.

## 4. Xem tiến độ và kết quả

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = 'C:\Users\Van\backdooranalog\runs\mmfi_p3s3_three_attacks_s42_4090_v1'
Get-ChildItem -LiteralPath $R -Filter 'controller.*.stderr.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Tail 30 }
foreach ($Method in 'input_aware','badnets','sig_source') {
    $Log = Join-Path $R ($Method + '\console.log')
    if (Test-Path -LiteralPath $Log) {
        Write-Host ('=== ' + $Method + ' ===')
        Get-Content -LiteralPath $Log -Tail 8
    }
}
if (Test-Path -LiteralPath (Join-Path $R 'comparison.md')) {
    Get-Content -LiteralPath (Join-Path $R 'comparison.md') -Encoding UTF8
}
```

Mỗi method giữ `checkpoint.pt`, `console.log`, `evaluation.json` cùng
artifact/binding chi tiết trong thư mục riêng. Chỉ gọi ba hàng mới hoàn tất
khi Status/evaluation cho biết cả ba complete; một số có đủ file chưa chứng
minh đúng provenance. Các hàng Clean/A1/Smooth/TSBA lịch sử không tự được ghép
hay đổi binding vào bảng này; ghép bảng P3S3 sau khi audit cùng protocol.

Việc code được push lên GitHub không tự launch experiment trên máy Vân.
Luồng chạy chính là **clone/pull → Inspect → Prepare → CheckOnly → Launch**;
không cần builder, ZIP, checkpoint cũ hay chép env từ máy khác. Ghi lại commit
đã lấy và giữ source checkout cố định sau Prepare để gate/resume khớp.
