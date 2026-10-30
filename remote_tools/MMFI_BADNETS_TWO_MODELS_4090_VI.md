# BadNets-CSI / MM-Fi P1S1 / WiFlow + MetaFi++ / seed 42

Setup riêng cho **máy Vân, RTX 4090**: WiFlow trước, MetaFi++ sau;
**tối đa 1 worker GPU**, không chạy hai victim song song. Đây là hai hàng
BadNets bổ sung cho bảng chung MM-Fi **P1S1**, không phải PiW3D hoặc P3S3,
và không phải bảng nhiều seed đã hoàn tất.

Chỉ chạy BadNets. Không tự train Clean, A1, TSBA, Smooth LF hay defense.
Không xóa/chuyển checkpoint cũ, không kill process, không đổi môi trường.

## 1. Lấy code ở checkout riêng trên máy Vân

Chạy PowerShell trên **máy Vân**, không phải máy 3090. Giữ nguyên các checkout
`C:\Users\Van\code` và `code_trojannn` của thí nghiệm trước; lấy bản này vào
`C:\Users\Van\code_badnets_mmfi`. Chỉ pull khi thư mục này sạch và đang ở main.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code_badnets_mmfi'
    if (Test-Path -LiteralPath $Repo) {
        if (!(Test-Path -LiteralPath "$Repo\.git")) { throw 'Thu muc ton tai nhung khong phai checkout; dung.' }
        $Branch = git -C $Repo branch --show-current
        if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc branch; dung.' }
        if ($Branch.Trim() -ne 'main') { throw 'Checkout khong o main; dung, khong tu doi branch.' }
        $Dirty = git -C $Repo status --porcelain
        if ($LASTEXITCODE -ne 0) { throw 'Git status loi; dung.' }
        if ($Dirty) { throw 'Checkout co thay doi; dung, khong ghi de.' }
        git -C $Repo pull --ff-only
    } else {
        git clone --depth 1 --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git $Repo
    }
    if ($LASTEXITCODE -ne 0) { throw 'Lay code loi; chua Prepare/Launch.' }
    git -C $Repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc commit; dung.' }
}
```

Nếu máy chưa có lệnh `git`, dừng và báo lỗi; helper không tự cài phần mềm.

## 2. Prepare riêng, chưa train full

Dùng Python/CUDA hiện có, không activate hay cài thêm môi trường.
Data full cần có ở `C:\Users\Van\backdooranalog\datasets\Compress`, cùng
`actions\data_bend.npy`. Prepare audit đủ TRAIN **133056**, TEST **33264**,
kiểm tra action/source/config và thực hiện CUDA smoke B32 cho từng backbone.
Kiểm tra lỗi/trường hợp thiếu dữ liệu trước khi Launch; không tự lấy subset.

```powershell
$S = 'C:\Users\Van\code_badnets_mmfi\remote_tools\Start-MMFiBadNetsTwoModels4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua Launch.' }
```

`PREPARED_ONLY` không phải kết quả train full. Local tests không thay cho audit
dữ liệu thật/CUDA/VRAM trên máy Vân; chưa có kết quả efficacy mới từ setup này.

## 3. Launch, tuần tự một job

Sau khi Prepare đã pass và không còn GPU compute khác, chạy riêng:

```powershell
$S = 'C:\Users\Van\code_badnets_mmfi\remote_tools\Start-MMFiBadNetsTwoModels4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
if ($LASTEXITCODE -ne 0) { throw 'Launch loi; xem log, khong tu retry.' }
```

**Giữ SSH mở**: Launch chạy foreground, không dùng tmux/Task Scheduler/WMI/
Start-Process ngầm; không bảo đảm sống sau khi ngắt SSH hoặc logoff. Log
stdout controller duy nhất cho mỗi lần gọi được in ở `CONTROLLER_LOG=...`;
stderr vẫn hiện trực tiếp trên cửa sổ SSH. Log worker chi tiết do runner lưu.
Nếu có process compute khác, dừng trước khi train; không bỏ qua guard, không tự
kill process của người khác. Các ứng dụng desktop WDDM đã nhận diện không được
đánh đồng với một Python compute job chưa rõ.

Idle guard là kiểm tra trạng thái tại thời điểm gọi và phối hợp giữa các launcher,
không phải cơ chế đặt chỗ GPU cứng: không ngăn người khác tự mở job không liên
quan sau đó. Cần phối hợp với người dùng máy, đợi compute hiện có kết thúc và
không mở thêm GPU job trong lượt chạy này; helper không tự dừng chúng.

WiFlow được train/evaluate xong rồi mới sang MetaFi++. Lỗi ở một phase dừng
hàng đợi, không tự giảm batch/đổi optimizer/AMP hoặc retry. Khi chạy lại sau khi
xác định lỗi, chỉ resume checkpoint tương thích đầy đủ, không fallback epoch 0
hay dùng checkpoint từ model/split/recipe khác. Không sửa file gate hoặc xóa lock
để ép chạy.

## 4. Xem tiến độ và bảng kết quả

Mở một SSH khác để xem; không đóng cửa sổ đang Launch.

```powershell
$S = 'C:\Users\Van\code_badnets_mmfi\remote_tools\Start-MMFiBadNetsTwoModels4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Xem log controller gần nhất (đọc, không ghi):

```powershell
$R = 'C:\Users\Van\backdooranalog\runs\mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1'
Get-ChildItem -LiteralPath "$R\logs" -Filter 'controller.*.log' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 20 }
```

PID sống hoặc file tồn tại không chứng minh train đang tiến triển: cần epoch/log
thay đổi và receipt hoàn tất. Xuất lại bảng từ kết quả đã có, không train mới:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Export
```

## Recipe và cách báo cáo

| Victim | Optimizer | LR | Physical batch | Accumulation / effective batch |
| --- | --- | --- | --- | --- |
| WiFlow | AdamW | 0.0001 | 32 | 2 / 64 |
| MetaFi++ | SGD, momentum 0.9 | 0.01 | 32 | 1 / 32 |

**SGD cho MetaFi++ ở MM-Fi là recipe hiện hành của bảng chung**, không phải
checkpoint SGD PiW3D đã loại trước đây. Không chuyển recipe này thành AdamW chỉ
vì nhánh Person-in-WiFi-3D dùng AdamW. WiFlow giữ ordinary MPJPE của comparison,
không thêm native bone loss/scheduler để thay đổi điều kiện sau khi xem điểm.

Cả hai scratch, seed42, 50 epochs/final49, rho0.1 uniform, cùng lịch dose
`0,.2,.4,.6,.8,1`, target joints `[2,3]`, pivot1, xoay Z `40*d` độ. Giữ nguyên
operator BadNets và giới hạn/numeric parameters từ comparison đã khóa; không
claim equal-noise giữa patch và các operator khác.

Bảng giữ T-MPJPE@1, Target-PCK@0.2@1, mean-positive DCE, Clean MPJPE và Clean
PA-MPJPE. Target-PCK chấm với attacker target và scale clean GT, không phải
Clean PCK/ASR. DCE là trung bình năm dose dương, không gộp dose0.
Same-victim no-trigger target controls được giữ trong artifact.

Không có checkpoint Clean reference tương thích trong lượt chạy hai hàng này:
clean-trained + trigger controls được đánh dấu **unavailable**, không mượn
Clean HPE-Li làm reference cho WiFlow/MetaFi++, không tự thêm hai job Clean.
Do đó báo đây là bổ sung hai hàng single-seed, không phải toàn bộ T1 hay
mean +/- độ lệch chuẩn giữa các seed.
