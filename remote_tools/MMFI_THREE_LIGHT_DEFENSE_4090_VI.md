# Defense A1 ba model MM-Fi trên RTX 4090 máy Vân

Bộ v3 chạy **HPE-Li → MetaFi++ → WiFlow**, mỗi lúc chỉ một job trên RTX 4090. Sáu phương pháp của từng model cùng xuất phát từ checkpoint A1 seed 42 tại epoch 49 **của chính model đó**: NoDef, fine tuning (FT), CLP, BNP, FST thích nghi cho hồi quy pose và Neural Cleanse (NC) thích nghi. BadNets và Fine-Pruning (FP) không nằm trong bộ này. Đây là so sánh **A1-only, một seed**, không phải bảng chứng minh defense chống nhiều loại attack.

FT dùng 5% TRAIN sạch được chọn cố định, 10 epoch, SGD `lr=1e-4`, batch vật lý 32. CLP dùng ngưỡng `u=3` trên trọng số; BNP dùng thống kê BatchNorm trên cùng TRAIN sạch và ngưỡng `u=3`. FST khởi tạo lại đầu tọa độ, fine tune 10 epoch và **chiếu chuẩn trọng số sau mỗi bước cập nhật optimizer**. Hạng phạt tích vô hướng ở bản pose được chuẩn hóa theo chuẩn đầu tọa độ và loss TRAIN sạch; đó là thay đổi so với tích vô hướng thô của bài FST gốc. Riêng MetaFi++, `decode.3` còn đi qua BatchNorm2d → ReLU → pooling → BatchNorm1d trước đầu ra pose, nên đây không phải phép chuyển trực tiếp một classifier tuyến tính. NC là phép thích nghi từ bài phân loại ảnh sang CSI/pose; nếu bước đảo trigger không qua điều kiện chỉ trên TRAIN, chỉ xuất chẩn đoán và **không** tạo hàng NC repair. Mọi repair bắt đầu từ cùng checkpoint A1 gốc của model đó. Không chọn cấu hình bằng TEST.

## Dữ liệu và code trên máy Vân

- Checkout mới nhất của repo tại `C:\Users\Van\code`, có `ATKBackd` và `remote_tools`. Python tại `C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.
- MM-Fi tại `C:\Users\Van\backdooranalog\datasets\Compress`; action prior gốc tại `C:\Users\Van\backdooranalog\actions\data_bend.npy`.
- HPE-Li: run Clean/A1 gốc và `three_tables.resolved.json`, mặc định `C:\Users\Van\backdooranalog\runs\mmfi_a1_three_tables_s42_43_44_v1`. Nếu khác, truyền `-HPELiSourceRoot` vào các lệnh.
- MetaFi++: archive `metafiplusplus_clean_a1_s42_20261024_164753.tar.gz` trong `C:\Users\Van\Downloads`, SHA256 `BB207177DFDEB232562BEA09D91B758BA47D8C41BCB5F7ACB38D256C63C89CE7`. Có thể dùng nguồn đã nhập đầy đủ tại `...\imports\metafiplusplus_s42_snapshot_20261025`; truyền `-MetaFiSourceRoot` nếu ở nơi khác.
- WiFlow: archive `wiflow_s42_snapshot_20261025_101131.zip` trong `C:\Users\Van\Downloads`, SHA256 `B2C92D6FE20E992F42A7E19D514DBEB0FC43F5D91C1C58B19860010EC0181B03`. Giữ ZIP để ghi dấu nguồn ngay cả khi đã giải nén; truyền `-WiFlowSourceRoot` nếu thư mục nhập ở nơi khác.

Archive chỉ mang code, manifest và checkpoint cần xác thực; dữ liệu MM-Fi vẫn lấy từ thư mục dataset trên máy Vân. Preflight kiểm tra hash và đúng Clean/A1 final49; nếu chỉ có checkpoint rời mà thiếu manifest/kết quả gốc, nó dừng.

## Chạy trên PowerShell của máy Vân

```powershell
$S = "$env:USERPROFILE\code\remote_tools\Start-MMFiThreeLightDefenses4090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

`Prepare` chạy dry run, kiểm tra nguồn/dữ liệu và smoke CUDA cho **sáu phương pháp của từng model**; nó chưa chạy defense đầy đủ. Chỉ khi cả ba preflight và GPU smoke PASS, `Launch` mới tạo Scheduled Task chạy tuần tự. Nếu bước nào lỗi, dừng tại đó và đọc log; không tự giảm batch hay chạy lại. Không khởi chạy job GPU khác cùng lúc. Task Scheduler dùng phiên đăng nhập Windows hiện tại; đóng terminal SSH/VS Code không phải lệnh dừng task, nhưng đăng xuất hoặc reboot có thể làm gián đoạn task.

Theo dõi:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = "$env:USERPROFILE\backdooranalog\runs"
Get-ChildItem "$R\*_light_a1_defense4090_s42_v3_bnp_fst_nc.task.*.stdout.log" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
  ForEach-Object { Get-Content -LiteralPath $_.FullName -Tail 25 -Wait }
```

Sau khi task kết thúc, xác thực kết quả và xuất bảng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\code\remote_tools\Collect-MMFiThreeLightDefense4090.ps1"
```

Mỗi model ghi vào `runs\<model>_light_a1_defense4090_s42_v3_bnp_fst_nc`. Bảng ghép nằm tại `runs\mmfi_three_light_a1_defense_report_s42_v3_bnp_fst_nc\table4_a1_only.md`. Bảng kiểm cả Clean MPJPE 17 khớp, Clean MPJPE trên **hai khớp mục tiêu**, và triggered-to-GT trên đúng hai khớp đó. Chênh lệch triggered-to-GT trừ clean-target mới là recovery gap cùng đơn vị/khớp; Target-PCK thấp một mình không chứng minh pose đã được phục hồi. Bảng NC chỉ xuất nếu có model sửa thật và evaluation hợp lệ.

Khi chỉ cần một model, ví dụ WiFlow, dùng `Start-MMFiLightDefense4090.ps1 -Backbone wiflow -Mode Prepare`, rồi chạy lại với `-Mode Launch`. Không chạy song song với task ba model.

Bài báo gốc để trích dẫn: [BNP / Pre-activation Distributions Expose Backdoor Neurons (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html) và [FST / Towards Stable Backdoor Purification through Feature Shift Tuning (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html). Kết quả MM-Fi ở đây phải ghi là **bản thích nghi cho hồi quy pose**, không phải tái lập nguyên bản hai bài đó.
