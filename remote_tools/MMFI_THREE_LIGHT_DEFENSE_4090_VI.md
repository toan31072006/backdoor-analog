# Defense A1 ba model MM-Fi trên RTX 4090 máy Vân

Bộ mới chạy **HPE-Li → MetaFi++ → WiFlow**, lần lượt một job trên GPU. Mỗi
backbone đánh giá NoDef, FT 10 epoch trên 5% TRAIN sạch (B32, SGD 1e-4), CLP
`u=3`, rồi Neural Cleanse thích nghi cho CSI/pose. Chỉ dùng A1 seed 42, checkpoint
victim epoch 49; không chạy lại victim hoặc dùng BadNets. ANP, NAD và FT-SAM
không thuộc bộ nhẹ này. Đây là pilot mới, không thay tên kết quả defense cũ.

## Đầu vào cần có trên máy Vân

- Code GitHub được pull vào `C:\Users\Van\code` (thư mục chứa `ATKBackd` và
  `remote_tools`). Python: `C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.
- Dataset MM-Fi tại `C:\Users\Van\backdooranalog\datasets\Compress`, prior gốc
  tại `C:\Users\Van\backdooranalog\actions\data_bend.npy`.
- Run HPE-Li gốc có `three_tables.resolved.json`, Clean/A1 final49 và đánh giá
  đầy đủ, mặc định ở `...\runs\mmfi_a1_three_tables_s42_43_44_v1`. Nếu ở chỗ
  khác, thêm `-HPELiSourceRoot '<đường dẫn>'` vào mọi lệnh dưới đây.
- Nếu MetaFi++ đã được giải nén đầy đủ vào
  `C:\Users\Van\backdooranalog\imports\metafiplusplus_s42_snapshot_20261025`, dùng lại
  nguồn đó; không cần tải tar lần nữa. Nếu chỉ có checkpoint riêng lẻ, cần cả
  manifest và kết quả gốc để xác thực nguồn; có thể chép
  `metafiplusplus_clean_a1_s42_20261024_164753.tar.gz` vào Downloads. Nếu
  nguồn đầy đủ nằm ở chỗ khác, thêm `-MetaFiSourceRoot '<đường dẫn>'` vào
  lệnh Prepare, Launch, Status và CheckOnly.
- Chép `wiflow_s42_snapshot_20261025_101131.zip` vào `C:\Users\Van\Downloads`.
  SHA256 WiFlow: `B2C92D6FE20E992F42A7E19D514DBEB0FC43F5D91C1C58B19860010EC0181B03`.
  SHA256 tar MetaFi++ (nếu cần): `BB207177DFDEB232562BEA09D91B758BA47D8C41BCB5F7ACB38D256C63C89CE7`.
  Runner tự xác thực hash trước khi nhập archive. Dữ liệu MM-Fi không nằm trong
  hai archive này.

## Chạy trong PowerShell trên máy Vân

```powershell
$S = "$env:USERPROFILE\code\remote_tools\Start-MMFiThreeLightDefenses4090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

`Prepare` gồm dry run, audit nguồn/dữ liệu và smoke CUDA thật cho từng phương
pháp, chưa sửa model đầy đủ. Nếu có lỗi, nó dừng ở đúng bước lỗi; không chạy
tiếp. `Launch` yêu cầu cả sáu receipt của ba model đều PASS rồi tạo một
Scheduled Task chạy tuần tự, bền hơn terminal SSH. Không chạy đồng thời script
defense cũ hoặc công việc Python khác trên GPU.

Theo dõi bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
Get-ChildItem "$env:USERPROFILE\backdooranalog\runs\*_light_a1_defense4090_s42_v2_nc.task.*.stdout.log" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
  ForEach-Object { Get-Content -LiteralPath $_.FullName -Tail 25 -Wait }
```

Sau khi task hoàn tất, chạy `-Mode CheckOnly`. Mỗi model có
`defense_results.md`, `defense_results.csv`, các record `evaluation.json`,
`per_sample.npz` và checkpoint riêng trong `runs\<model>_light_a1_defense4090_s42_v2_nc`.
`CheckOnly` xác thực file và tính lại metric từ mảng từng frame.

Xuất một bảng ghép cặp cho cả ba model bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\code\remote_tools\Collect-MMFiThreeLightDefense4090.ps1"
```

Bảng nằm trong `runs\mmfi_three_light_a1_defense_report_s42_v2_nc\table4_a1_only.md`.

Neural Cleanse gốc là phương pháp cho phân loại ảnh. Bản này thử tám giả
thuyết target pose, đảo trigger CSI trên TRAIN và kiểm bằng Clean model trên
TRAIN. Nếu không khôi phục được trigger theo ngưỡng đã cố định, chỉ có
`nc_diagnostic.json`; **không có hàng NC repair trong bảng**. Nếu đạt ngưỡng,
runner unlearn một epoch rồi đo lại toàn bộ TEST. Bảng cần đọc cả Clean MPJPE
17 khớp, Clean MPJPE 2 khớp mục tiêu và triggered-to-GT 2 khớp; Target-PCK
thấp một mình không chứng minh pose đã được khôi phục. Chỉ có một seed, không
báo SD/CI giữa các seed hoặc khẳng định chống được attack khác A1.
