# HPE-Li trên RTX 4090: hai attack MM-Fi P1S1, seed 42

Pilot riêng gồm **TSBA-adapted** và **smooth-LF-CSI control**, mỗi phương pháp
train một HPE-Li mới từ đầu. Dữ liệu MM-Fi protocol1-s1, target chân [2,3],
rho=.1, batch32, checkpoint cuối epoch49 và sáu dose giữ theo run MM-Fi seed42.
Hai job cùng chạy trên `cuda:0` **chỉ sau khi GPU smoke đồng thời của chính hai
job đạt**. Output mới ở `runs\mmfi_hpeli_extra_attacks_s42_4090_v1`; không ghi
vào các run MetaFi++, WiFlow, PiW3D hoặc HPE-Li cũ.

LF ở đây dùng pattern DCT CSI cố định lấy ý tưởng từ LF, không phải bản tái
hiện pattern tối ưu của LF gốc. TSBA cần gradient qua victim khi học generator,
vì vậy hai phương pháp khác nhau về quyền attacker và biên độ trigger. Một seed
không cho phép ước lượng độ dao động giữa các seed.

## Máy Vân: chuẩn bị và kiểm trước khi train

Mở PowerShell trên máy RTX 4090, bảo đảm repo đã được cập nhật và GPU không còn
job khác. Dữ liệu ở `%USERPROFILE%\backdooranalog\datasets\Compress`, action ở
`%USERPROFILE%\backdooranalog\actions\data_bend.npy`, Python ở
`%USERPROFILE%\backdooranalog\envs\dose-backdoor\python.exe`.

```powershell
cd "$env:USERPROFILE\code"
git pull --ff-only
$S = "$env:USERPROFILE\code\remote_tools\Start-MMFiHPELiExtraAttacks4090.ps1"
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Preflight
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode GpuSmoke
```

`DryRun` chỉ in kế hoạch; `Preflight` kiểm dữ liệu/source mà không train;
`GpuSmoke` chạy **hai model thật cùng lúc** bằng batch32 và kiểm bộ nhớ 4090.
Nếu một bước lỗi, dừng tại đó và gửi log. Không tự giảm batch hoặc launch khi
thiếu `prelaunch_gate.json`. Kiểm tra kết quả smoke trước khi launch:

```powershell
$R = "$env:USERPROFILE\backdooranalog\runs\mmfi_hpeli_extra_attacks_s42_4090_v1"
Get-Content "$R\prelaunch_gate.json" -Encoding UTF8
```

## Launch sau khi GPU smoke đạt

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Launcher dùng `Start-Process -WindowStyle Hidden`; không cần quyền admin hoặc
Scheduled Task. Nó in PID và đường dẫn controller stdout/stderr. Dòng `STARTED`
chỉ xác nhận process được tạo; muốn xác nhận tiến độ, xem log:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
Get-ChildItem "$R\logs\*.log" -ErrorAction SilentlyContinue |
  ForEach-Object { "===== $($_.BaseName) ====="; Get-Content $_.FullName -Tail 8 -Encoding UTF8 }
```

Giữ phiên Windows đăng nhập và máy bật trong lúc train. Nếu một job lỗi,
controller giữ checkpoint/log để chẩn đoán; không tự chạy lại hoặc dừng job khác.
Sau khi cả hai xong, xác minh checkpoint cuối và evaluation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
```

Chỉ dùng kết quả khi `CheckOnly` xác nhận cả **2/2**. GPU smoke local và train
thật trên máy Vân là hai bước khác nhau; thời gian/VRAM thực tế chỉ biết sau
khi chạy `GpuSmoke` và xem những epoch đầu.
