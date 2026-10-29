# PiW3D / MetaFi++ AdamW: chạy nốt trên RTX 3090

Chỉ **một worker trên `cuda:0`**, chạy tuần tự sáu hàng: Clean reference,
Proposed A1, FTrojan-CSI, BppAttack-CSI, FIBA-CSI và WaNet-CSI. **Không chạy TSBA
adapted hoặc Smooth LF-CSI**. Đây là nhánh Person-in-WiFi-3D, không phải MM-Fi.

Code lấy bằng GitHub; checkpoint là dữ liệu riêng, không đưa lên Git. Không
phải chuyển một ZIP code thủ công. Launcher không cài môi trường, không dừng
process khác, không sửa hoặc xóa run lịch sử.

## 1. Lấy code trên máy 3090

Chạy trong **PowerShell máy 3090**, không phải terminal Linux/Vast. Nếu repo đã
có ở `C:\Users\ssh3090\code`, dùng đường dẫn đó. Giữ mọi thay đổi local; nếu
`git pull` báo file untracked/dirty thì dừng, không dùng reset hoặc xóa file.

```powershell
$Repo = 'C:\Users\ssh3090\code'
git -C $Repo remote get-url origin
git -C $Repo status --short
git -C $Repo pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; dừng.' }
$S = Join-Path $Repo 'remote_tools\Start-PiWMetaFiRemaining3090.ps1'
```

Origin phải là `https://github.com/toan31072006/backdoor-analog.git` (hoặc SSH
cùng repo). Nếu chưa có clone, dùng một thư mục mới, không ghi đè code đang chạy:

```powershell
$Repo = 'C:\Users\ssh3090\code-piw-metafi-remaining'
if (Test-Path -LiteralPath $Repo) { throw 'Đường dẫn đã tồn tại; chọn thư mục mới.' }
git clone https://github.com/toan31072006/backdoor-analog.git $Repo
if ($LASTEXITCODE -ne 0) { throw 'Clone lỗi; dừng.' }
$S = Join-Path $Repo 'remote_tools\Start-PiWMetaFiRemaining3090.ps1'
```

## 2. Dữ liệu và checkpoint trên máy 3090

Mặc định launcher dùng:

```text
C:\Users\ssh3090\backdooranalog\
  datasets\Person-in-WiFi-3D_processed_v1\
    train_data\train_data_list.txt
    test_data\test_data_list.txt
  actions\data_bend.npy
  imports\piw_metafi_adamw_unfinished\
    metafiplusplus\seed42\
      ftrojan\checkpoint.pt
      bpp_source\checkpoint.pt
      fiba\checkpoint.pt
  runs\piw_metafi_remaining_s42_3090_v1\
```

Ba checkpoint nhập là **AdamW đang dở**, không phải checkpoint SGD cũ hoặc
checkpoint đã có kết quả hoàn chỉnh. Import chỉ được dùng khi SHA256, model,
epoch, optimizer/RNG, cấu hình và provenance khớp nguồn đã lưu. Không sửa hash
hoặc đổi tên tùy ý để vượt kiểm tra. Controller còn tìm trong
`DataHome\runs\**\attacks\metafiplusplus\seed42\<method>\checkpoint.pt` và
CheckpointHome. Chỉ nhận checkpoint **AdamW cùng scientific plan**; checkpoint
SGD không được nhập. Nếu có nhiều checkpoint hợp lệ, chọn **epoch cao nhất**,
không chọn theo điểm TEST. Checkpoint epoch199 chỉ evaluate, không train thêm.
Controller cũng tìm các file backup đã chuyển ở
`DataHome\code\**\recovery_inputs\ftrojan_epoch84.pt`,
`bpp_source_epoch72.pt` và `fiba_epoch33.pt`. Nếu các file đúng SHA đã ở đó,
**không cần chuyển lại 911 MiB**; Inspect sẽ liệt kê vị trí tìm thấy.

Clean/A1/WaNet chỉ bắt đầu mới nếu không tìm được checkpoint AdamW cùng plan ở
các vị trí trên. Không tự nhìn thấy checkpoint trên máy Vân: muốn reuse phải
chuyển cả checkpoint và plan gốc đầy đủ sang đúng cấu trúc trên. Không tự fallback
scratch cho FTrojan/BPP/FIBA khi thiếu checkpoint đang dở; phải chuyển backup
gốc vào rồi Prepare lại.

| Method | Epoch đã lưu (từ 0) | SHA256 checkpoint gốc |
| --- | ---: | --- |
| FTrojan-CSI | 84 | `9d2a713315f694183cd9512e9feb74fe4f44bb823fedf455fafbf5edddd7d093` |
| BppAttack-CSI | 72 | `4ec9d9ff57fa578a42f368764b9c110d2376e155d6d578d5831676a5fd2a4c5f` |
| FIBA-CSI | 33 | `590394ff220b1b99dcd32db6416948e40fa6b49fb4f8934f7dfd9f3bbda98389` |

Chúng nằm trên máy mình tại
`E:\wifi\deliverables\vast_adamw_unfinished_20261028\metafiplusplus\seed42`;
tổng ba file 954,888,813 bytes (khoảng 911 MiB). Không chuyển Smooth LF epoch45.

Nguồn code khoa học nhỏ, đã cố định và lưu trong Git, được kiểm SHA trước khi
dùng. Đây là snapshot phục vụ resume, không phải yêu cầu user chuyển ZIP code.
Giữ nguyên các file nguồn/plan đi kèm sau Prepare; pull thay code giữa Prepare
và Launch có thể làm gate không còn hợp lệ.

Ba checkpoint backup đã pin SHA ở trên có plan gốc đóng kèm trong Git nên không
cần tự chuyển thêm `plan.json` cho chính ba file đó. Checkpoint AdamW khác từ
Vân/3090 vẫn cần plan gốc đầy đủ, không dùng plan của file khác thay thế.

FTrojan từng gặp khác biệt pattern Linux/Windows. Bản này giữ pattern Linux gốc
đã khôi phục: SHA256 `812754c588d835a3e19f503488237e334dee0bd4c17c39404597504a78b25e45`,
asset nén 2,385 bytes đi qua Git. Pattern được kiểm đúng với checkpoint trước
resume; không đổi trigger hoặc sửa checkpoint gốc để vượt mismatch.

Checkpoint và dataset không tự có trên 3090 chỉ vì đã pull. Kiểm tra trước:

```powershell
$H = 'C:\Users\ssh3090\backdooranalog'
Get-Item -LiteralPath "$H\datasets\Person-in-WiFi-3D_processed_v1\train_data\train_data_list.txt",
    "$H\datasets\Person-in-WiFi-3D_processed_v1\test_data\test_data_list.txt",
    "$H\actions\data_bend.npy"
Get-ChildItem -LiteralPath "$H\imports\piw_metafi_adamw_unfinished" -Recurse -File -Filter checkpoint.pt |
    Select-Object FullName,Length | Format-List
```

Nếu dùng dataset, motion prior, môi trường hoặc checkpoint ở nơi khác, truyền
cùng các tham số cho **tất cả** lần Inspect/Prepare/Launch/Status:

```powershell
$RunOptions = @{
    DataHome = 'C:\Users\ssh3090\backdooranalog'
    DatasetRoot = 'D:\datasets\Person-in-WiFi-3D_processed_v1'
    ActionNpy = 'C:\Users\ssh3090\backdooranalog\actions\data_bend.npy'
    CheckpointHome = 'D:\imports\piw_metafi_adamw_unfinished'
    PythonExe = 'C:\Users\ssh3090\backdooranalog\envs\piw-metafi\Scripts\python.exe'
}
```

Nếu dùng mặc định, đặt `$RunOptions = @{}`. Launcher tìm Python sẵn có theo thứ
tự `envs\piw-metafi\Scripts\python.exe`, `envs\dose-backdoor\python.exe`, rồi
`envs\mmfi-p3\Scripts\python.exe`. Không cài hay thay dependency của môi trường
bạn khác; runtime không tương thích phải được xử lý trước full training.

Constructor MetaFi++ cần cache pretrained ResNet34 ImageNet gốc, kể cả trước khi
load trọng số đã resume. Controller tự tìm cache đúng SHA trong
`DataHome\torch-cache`, `USERPROFILE\.cache\torch` và
`DataHome\code\**\torch-cache`; nếu không có mới để torchvision tải từ nguồn
chính thức. Có thể đặt `$env:TORCH_HOME` tới thư mục cache chứa
`hub\checkpoints\resnet34-b627a593.pth`, giữ cùng giá trị khi Prepare/Launch.
Không đưa cache lên Git hoặc chuyển sang random init để tránh thiếu cache.

### Chuyển trọng số từ máy mình

Thực hiện trong **PowerShell máy mình (`E:\wifi`)**, không phải bên máy 3090.
Chỉ cần chuyển nếu Inspect chưa tìm thấy backup. SSH config máy mình đã có alias
**`RTX-3090`**, hostname `100.98.82.67`, user `ssh3090` (không phải alias
`ssh3090`). Kết nối không tương tác hiện chưa xác thực được; lệnh `scp` có thể
cần bạn nhập mật khẩu hoặc hoàn tất SSH key, không có nghĩa file đã được chuyển.

Tạo thư mục đích bằng **PowerShell trên 3090**:

```powershell
$I = 'C:\Users\ssh3090\backdooranalog\imports\piw_metafi_adamw_unfinished\metafiplusplus\seed42'
foreach ($m in 'ftrojan','bpp_source','fiba') {
    New-Item -ItemType Directory -Force -Path (Join-Path $I $m) | Out-Null
}
```

Sau đó trên **PowerShell máy mình**:

```powershell
$Source = 'E:\wifi\deliverables\vast_adamw_unfinished_20261028\metafiplusplus\seed42'
$Target = 'RTX-3090'
$Remote = 'C:/Users/ssh3090/backdooranalog/imports/piw_metafi_adamw_unfinished/metafiplusplus/seed42'
foreach ($m in 'ftrojan','bpp_source','fiba') {
    $File = Join-Path $Source "$m\checkpoint.pt"
    Get-FileHash -LiteralPath $File -Algorithm SHA256
    scp $File "${Target}:$Remote/$m/checkpoint.pt"
    if ($LASTEXITCODE -ne 0) { throw "Chuyển $m lỗi; dừng." }
}
```

Đối chiếu SHA trên **máy 3090** sau khi chuyển:

```powershell
$I = 'C:\Users\ssh3090\backdooranalog\imports\piw_metafi_adamw_unfinished\metafiplusplus\seed42'
foreach ($m in 'ftrojan','bpp_source','fiba') {
    Get-FileHash -LiteralPath (Join-Path $I "$m\checkpoint.pt") -Algorithm SHA256
}
```

Không gửi checkpoint lên GitHub. Không dùng `scp` với đường dẫn `E:\...` bên
trong terminal Linux; không chuyển cả thư mục backup vì nó có Smooth LF bị loại.

## 3. Inspect rồi Prepare, chưa chạy full

```powershell
$RunOptions = @{}
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect @RunOptions
if ($LASTEXITCODE -ne 0) { throw 'Inspect lỗi; dừng.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare @RunOptions
if ($LASTEXITCODE -ne 0) { throw 'Prepare lỗi; chưa chạy full.' }
```

Inspect chỉ đọc tình trạng. Prepare kiểm đầy đủ source/dữ liệu/checkpoint,
thực hiện bước CUDA **physical batch 32**, lưu gate rồi **dừng**. Prepare có thể
tốn thời gian đọc toàn bộ dữ liệu; không đồng nghĩa đã train 200 epoch. Chỉ
launch khi đã thấy `PREPARED_ONLY`. Nếu GPU đang có job khác, không tắt bừa
process hoặc bỏ guard để chạy chồng.

## 4. Launch riêng sau Prepare đạt

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch @RunOptions
if ($LASTEXITCODE -ne 0) { throw 'Launch lỗi; xem log.' }
```

Controller chạy ẩn, có log riêng; ngắt SSH không chủ động hủy controller. Không
tắt máy, sleep hoặc đóng process Python. Một PID đang sống không chứng minh
epoch đang tiến lên; kiểm log. Launch lại có kiểm gate và lock, không tạo thêm
worker song song cho cùng output.

Sáu victim chạy tuần tự tới final199, sau đó evaluate lần lượt cùng final Clean
reference. Hàng có checkpoint final199 không train thêm. Artifact hoàn tất chỉ
được skip sau kiểm tra hợp lệ, không chỉ thấy một tên file.
Một worker lỗi thì hàng đợi dừng; không tự retry, giảm batch, bật AMP hoặc chạy
tiếp các hàng sau để che lỗi. Giữ checkpoint epoch gần nhất, không ghi đè
checkpoint nhập ban đầu.

## 5. Kiểm tra và chạy tiếp

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status @RunOptions
$R = 'C:\Users\ssh3090\backdooranalog\runs\piw_metafi_remaining_s42_3090_v1'
Get-ChildItem -LiteralPath $R -File -Filter 'controller.*.stdout.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 25 }
Get-ChildItem -LiteralPath $R -File -Filter 'controller.*.stderr.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 25 }
nvidia-smi
```

Khi controller cũ đã thoát và không còn worker của run đang chạy, gọi lại
`-Mode Launch` với **cùng code, môi trường, input và RunOptions** để tiếp tục
từ checkpoint hợp lệ gần nhất. Nếu gate thay đổi thì phải xem lý do; không xóa
gate/lock hay sửa checkpoint để ép chạy. Không cần train lại hàng đã hoàn tất.

## Phạm vi khoa học và giới hạn

- Person-in-WiFi-3D one-person, seed 42; full ordered TRAIN 27,876 và TEST
  2,586, cùng filtering/normalization/14-joint pose của nhánh đã lưu.
- MetaFi++ nhánh PiW/DT-Pose, **AdamW lr=.01, weight decay=.01**, physical batch 32, victim 200 epoch,
  cuối epoch 199; không chuyển sang SGD, không dùng checkpoint SGD.
- Poison rho .1, fixed paired doses U(.2,1); target body-hip-axis v2: pivot7,
  khớp [11,13], quay 40*d độ quanh trục `normalize(clean_GT[5]-clean_GT[7])`,
  PCK scale clean GT(6,4), threshold .2. Không sửa A1, target hoặc native
  operator để cải thiện điểm.
- Evaluation đủ sáu dose 0,.2,.4,.6,.8,1; T-MPJPE và Target-PCK tại dose 1 trên
  hai khớp mục tiêu, Clean MPJPE/PA toàn bộ khớp, DCE trung bình năm dose dương.
  Giữ no-trigger/clean controls và per-sample artifact, không gọi Target-PCK là
  clean PCK hoặc ASR.
- Một seed là n=1, không có across-seed SD. Sáu hàng là phạm vi khôi phục mới
  user yêu cầu, **không phải bảng tám hàng hoàn chỉnh**: TSBA/Smooth LF bị loại
  có chủ đích ở batch này, không xóa kết quả lịch sử của chúng.
- Chuyển checkpoint giữa Linux/Windows hoặc GPU khác không bảo đảm bitwise
  cùng quỹ đạo số. Provenance import/runtime mới được ghi riêng, không nhận là
  một lần resume trên nguyên host. CPU/local tests không chứng minh tốc độ,
  VRAM RTX 3090 hoặc hiệu quả attack; cần Prepare CUDA trên máy thật.
- Không có ETA đáng tin chỉ từ tên GPU/epoch; đo thời gian vài epoch mới sau
  khi chỉ còn một worker. Không tự chạy defense trong batch train này.
