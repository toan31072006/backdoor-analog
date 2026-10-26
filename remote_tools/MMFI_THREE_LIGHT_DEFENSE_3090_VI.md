# MM-Fi P1S1 seed42: defense trên RTX 3090

Phạm vi: ba victim **HPE-Li, MetaFi++, WiFlow** trên MM-Fi P1S1, attack **A1**.
Mỗi model chạy tuần tự NoDef, FT 10 epoch, CLP, BNP, FST adapted, Neural Cleanse adapted.
Chỉ một GPU job mỗi lúc. Đây là bảng mới **exploratory, một seed**; không gộp với
T5 ba seed/recipe cũ và không dùng checkpoint HPE-Li P3 trên máy 3090.

## Nguồn cần chuyển

- Máy Vân: run gốc `C:\Users\Van\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1`
  chứa checkpoint **P1S1 final49** của HPE-Li Clean/A1. Chạy bằng Python của
  `dose-backdoor` sau khi lấy code mới:

  ```powershell
  $Py = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
  $Repo = "$env:USERPROFILE\code"
  if (!(Test-Path "$Repo\remote_tools\Export-MMFiHPELiP1S1-S42.py")) {
      $Repo = "$Repo\backdoor-analog"
  }
  & $Py "$Repo\remote_tools\Export-MMFiHPELiP1S1-S42.py"
  if ($LASTEXITCODE -ne 0) { throw 'HPE-Li export failed' }
  ```

  Lưu SHA256 mà lệnh in ra. Script chỉ đọc run gốc, xác minh manifest,
  checkpoint/evaluation hashes, rồi tạo file mới
  `Downloads\mmfi_hpeli_p1s1_s42_clean_a1_source.zip`.
- MetaFi++: `metafiplusplus_clean_a1_s42_20261024_164753.tar.gz`, SHA256
  `BB207177DFDEB232562BEA09D91B758BA47D8C41BCB5F7ACB38D256C63C89CE7`.
- WiFlow: `wiflow_s42_snapshot_20261025_101131.zip`, SHA256
  `B2C92D6FE20E992F42A7E19D514DBEB0FC43F5D91C1C58B19860010EC0181B03`.

Chép cả ba vào `C:\Users\ssh3090\backdooranalog\downloads`. File results-only
không chứa checkpoint, nên không thể thay một trong ba nguồn này.

## Trên RTX 3090

Code phải chứa `remote_tools\Start-MMFiThreeLightDefenses3090.ps1`. Đặt:

```powershell
$H = "$env:USERPROFILE\backdooranalog"
$S = "$H\code\backdoor-analog\remote_tools\Start-MMFiThreeLightDefenses3090.ps1"
$HPESHA = 'DAN_SHA256_IN_RA_TREN_MAY_VAN_VAO_DAY'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode DryRun
```

Nếu code tải qua GitHub ZIP, `$S` là đường dẫn trong thư mục đã giải nén.
Có thể tải và chuyển nguồn khi P3 còn chạy. `Preflight` dùng CPU/đĩa và
kiểm tra toàn bộ input; chạy sau khi đã đặt đủ ba archive:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Preflight -HPELiArchiveSHA256 $HPESHA
```

**Đợi đủ sáu P3 evaluation P3S1/S2/S3 Clean/A1 complete/final49** trước hai
lệnh sau. Runner chặn nếu thiếu một evaluation. GpuSmoke đo từng phương pháp
trên đúng RTX 3090, không tự giảm batch/đổi recipe khi thiếu VRAM:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode GpuSmoke
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode LaunchForeground
```

`LaunchForeground` phải chạy từ phiên Windows PowerShell độc lập và được giữ mở.
Không chạy nó trong terminal VS Code Remote: lần P3 trước, terminal bị đóng khi
VS Code server mất kết nối. Tài khoản `ssh3090` từng bị từ chối tạo Scheduled
Task; launcher này không giả vờ có cơ chế chạy nền bền vững. Nếu không có phiên
Windows độc lập, cần quản trị viên cấp quyền Scheduled Task trước khi chạy lâu.

Xem tình trạng và log:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
Get-Content "$H\runs\hpeli_light_a1_defense3090_s42_v1.controller.log" -Tail 20
Get-Content "$H\runs\metafiplusplus_light_a1_defense3090_s42_v1.controller.log" -Tail 20
Get-Content "$H\runs\wiflow_light_a1_defense3090_s42_v1.controller.log" -Tail 20
```

Mỗi output có `defense_results.csv/md`; metric gồm T-MPJPE, Target-PCK,
clean MPJPE toàn thân, **clean MPJPE hai khớp mục tiêu**, triggered-to-GT
hai khớp và DCE. Neural Cleanse có thể kết thúc ở trạng thái diagnostic-only;
trường hợp đó không được báo như một repair thành công.
