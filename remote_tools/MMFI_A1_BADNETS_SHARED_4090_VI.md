# Máy Vân: một A1 + một BadNets trên 4090

Đây là **execution-sharing được người dùng cho phép riêng**: tối đa một A1
motion-prior HPE-Li và một BadNets WiFlow/MetaFi++ trên `cuda:0`. Các prior A1
cross → nod vẫn tuần tự; hai backbone BadNets vẫn tuần tự. Không đổi optimizer,
batch, lịch epoch, operator, target, poison hay metric. GPU-util còn dư không
bảo đảm đủ VRAM hoặc tăng tốc; không retry, giảm batch, AMP hay đổi recipe nếu
CUDA smoke/training lỗi.

## 1. Tạm dừng queue BadNets cũ trước

Helper shared **không attach vào queue sống, không kill process**. Trong cửa sổ
SSH đang Launch BadNets, chờ log `[ckpt] saved` hoàn tất một epoch rồi bấm
**Ctrl+C** để ngắt queue. Checkpoint vừa lưu giữ model/optimizer/RNG để resume;
phần epoch chưa lưu có thể phải chạy lại. Giữ cửa sổ đó cho tới khi controller
và worker thoát. Không tắt SSH giữa lúc đang ghi checkpoint và không xóa lock.

Mở PowerShell thứ hai trên **máy Vân** để xác nhận không còn controller hoặc
worker của snapshot BadNets cũ:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match 'run_mmfi_badnets_two_models_4090\.py' } |
    Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine |
    Format-List
```

Phải **không còn dòng nào** trước Prepare. Nếu còn worker, chưa chạy helper
mới; chờ nó thoát và gửi output để kiểm tra. Không dùng lệnh kill mọi Python,
không tự xóa process dựa trên GPU-util thấp. Đồng thời không mở lại launcher
BadNets cũ khi shared controller đang chạy.

Checkpoint cũ nằm dưới:
`C:\Users\Van\backdooranalog\runs\mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1`.
Có thể xem metadata file và dòng save gần nhất, nhưng file tồn tại chưa chứng
minh checkpoint hợp lệ:

```powershell
$R = 'C:\Users\Van\backdooranalog\runs\mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1'
Get-ChildItem -LiteralPath $R -Recurse -File -Filter checkpoint.pt |
    Select-Object FullName,Length,LastWriteTime
Get-ChildItem "$R\logs\run.*.log" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Tail 12 -Encoding UTF8 }
```

Shared Prepare/Launch kiểm tra strict state/config/source/input/gate trước
resume. Checkpoint thiếu optimizer/RNG hoặc binding sai là lỗi phải dừng; không
tự quay về epoch0, không chép/đổi tên checkpoint từ profile khác. Backbone chưa
bắt đầu chỉ train fresh theo cell đã được chốt, không warm-start WiFlow sang
MetaFi++.

## 2. Code mới ở checkout riêng

Giữ nguyên **`C:\Users\Van\code_badnets_mmfi`**: không pull, thêm file, sửa
script hoặc Prepare lại snapshot cũ. Plan BadNets bind toàn bộ source ATKBackd.
Lấy code shared vào **`C:\Users\Van\code_motion_priors`**:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code_motion_priors'
    if (Test-Path "$Repo\.git") {
        if (git -C $Repo status --porcelain) { throw 'Checkout có thay đổi; dừng để tránh ghi đè.' }
        git -C $Repo pull --ff-only
    } else {
        git clone --depth 1 --branch main https://github.com/toan31072006/backdoor-analog.git $Repo
    }
    if ($LASTEXITCODE -ne 0) { throw 'Lấy code lỗi; chưa Prepare.' }
}
```

Tải action cross/nod và audit CPU theo
[hướng dẫn motion priors](MMFI_A1_MOTION_PRIORS_4090_VI.md). Downloader chỉ lấy
hai file chính thức khoảng441.9 MiB, dùng Python có sẵn, không gdown/install hay
training. Không tự duyệt file upload chưa rõ nhãn; giữ provenance sidecar.
Đọc [recipe motion-prior](MMFI_A1_MOTION_PRIORS_4090_VI.md#recipe-được-giữ-cố-định)
và [recipe BadNets](MMFI_BADNETS_TWO_MODELS_4090_VI.md) trước chạy.

Python mặc định: `C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.
Shared receipts/logs mới nằm ngoài cả hai source checkout:
`C:\Users\Van\backdooranalog\tools\mmfi_a1_badnets_shared_v1`.
Không cài environment mới và không sửa plan/gate/checkpoint lịch sử để đổi
execution. Receipt shared riêng ghi rõ gate cũ chỉ chứng nhận context cũ,
không chứng nhận concurrent context mới.

## 3. Prepare shared rồi Launch riêng

Chỉ sau khi old queue đã thoát hết và có đúng action/provenance:

```powershell
$S = 'C:\Users\Van\code_motion_priors\remote_tools\Start-MMFiA1BadNetsShared4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -AllowConcurrentBadNets
```

Prepare full audit, xác minh snapshot/checkpoint và CUDA B32 smoke trong
execution context được phép; **không train full victim**. Chỉ khi thành công,
kiểm tra receipts rồi gọi riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -AllowConcurrentBadNets
```

Launch kiểm tra gate trước, foreground có Tee stdout ra log mới. **Giữ SSH
mở**. Không có tmux/ScheduledTask/WMI/service, không tự tạo detached process.
Tối đa một own A1 + một own BadNets; unknown/compute jobs khác vẫn bị chặn.
Nếu một lane lỗi, không launch job kế tiếp hay kill lane còn lại: giữ SSH,
đợi owned peer thoát rồi đọc log. Không coi một worker lỗi là đã hoàn thành.
Trong **shared controller mới**, Ctrl+C cũng đi vào fail-stop/drain: không phát
tín hiệu dừng các worker đang thuộc quyền của nó, chờ chúng kết thúc rồi nhả
lock. Đây không phải nút pause ngay cả hai lane. Hướng dẫn Ctrl+C ở Mục1 chỉ
áp dụng **queue native cũ trước khi chuyển sang shared**, không phải phiên
shared đang chạy. Nếu đã bấm Ctrl+C trong shared, vẫn giữ SSH cho tới khi drain
xong; không mở lại một controller khác hay tự xóa lock trong lúc đó.

Không chạy đồng thời helper native exclusive `Start-MMFiA1MotionPriors4090.ps1`
hoặc launcher BadNets cũ. Shared helper đã sở hữu cả hai lane. Nếu chỉ muốn chạy
A1 exclusive, dùng hướng dẫn native sau khi BadNets hoàn tất, không trộn hai
execution profiles trong cùng output.

## 4. Kiểm tra tiến độ

Trong SSH thứ hai:

```powershell
$S = 'C:\Users\Van\code_motion_priors\remote_tools\Start-MMFiA1BadNetsShared4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

`Status` không cần switch cho phép và không launch training. Muốn check gate
riêng trước Launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly -AllowConcurrentBadNets
```

Log cell UTF-8; log Tee của Windows PowerShell5.1 có thể UTF-16. Xem dòng epoch
mới, mtime/log và authenticated completion cùng nhau; PID cũ hoặc file tồn tại
không phải bằng chứng tiến độ. Không suy ETA hai lane từ một snapshot GPU-util.

Đây là pilot seed42, không phải bảng multi-seed hoàn chỉnh. Bend lịch sử chưa
được nhập giả vào bảng; motion prior khác không đổi pose target. Local CPU/stub
tests không thay cho CUDA/full-data smoke trên Vân hoặc efficacy thực tế.
