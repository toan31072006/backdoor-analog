# TrojanNN adapted: HPE-Li / MM-Fi P1S1 / seed 42

Chỉ chạy **P1S1**, một job trên `cuda:0`. Không chạy P3S3, không đụng các
job A1/SIG/MetaFi hoặc kết quả lịch sử. Đây là một hàng bổ sung cho comparison,
không phải tuyên bố đã hoàn thành cả bảng hay đã có kết quả TrojanNN.

## Material Passport

- Origin: academic-research-suite / experiment-agent, 2026-10-29.
- Status: implementation/CPU verification; không phải full-data/CUDA result.
- Phạm vi được duyệt: code + test + push; người dùng tự Prepare/Launch.
- T1 extension riêng, chưa gộp/thay thế các hàng hoặc kết quả lịch sử.

## Điều kiện

- Code mới và môi trường CUDA đã có sẵn; helper không cài/sửa environment.
- MM-Fi đầy đủ tại `DataHome\datasets\Compress`, motion prior tại
  `DataHome\actions\data_bend.npy` (audit SHA gốc theo protocol comparison;
  TrojanNN không dùng motion prior này để tối ưu patch).
- Một checkpoint **Clean HPE-Li / MM-Fi P1S1 / seed42 / final49** khớp recipe.
  Mặc định dùng
  `C:\Users\Van\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1\seed42\clean\checkpoint.pt`.
  Nếu không có, chỉ truyền đường dẫn checkpoint đúng bằng `-CleanCheckpoint`;
  không lấy checkpoint A1, PiW3D, P3S3 hay backbone khác thay thế.
- GPU chưa chạy một job tính toán khác. Không tắt AnyDesk/TeamViewer để chạy;
  runner phân biệt ứng dụng desktop với công việc CUDA và chặn process chưa xác định.
  Helper không kill process, không có cờ bỏ qua guard.

TrojanNN là **model-informed adaptation**: đọc clean reference để chọn neuron
và tối ưu một trigger phổ quát. Sau đó đóng băng trigger, huấn luyện victim
poisoned **từ khởi tạo mới**, không fine-tune/warm-start victim từ clean reference.
Đây không phải threat model hoàn toàn không biết victim/model. Không gọi là
reproduction nguyên bản trên ảnh: CSI/HPE dùng input float, target pose và dose
đã khai báo; không thêm bước PIL/uint8 của pipeline ảnh.

## Hợp đồng comparison

P1S1 full split: **133056 TRAIN / 33264 TEST**, split seed0. HPE-Li seed42,
50 epoch/final49, physical batch32, SGD lr0.001/momentum0.9/weight-decay0,
FP32 và workers0 như recipe comparison hiện tại. Poison rate0.1, chọn uniform;
trigger/payload dose paired `U(0.2,1)`. Pose target giữ pivot1, joints[2,3],
xoay quanh Z tối đa40 độ theo dose; scale Target-PCK từ clean-GT joints[5,12].
Không sửa A1 hay recipe để ưu tiên một baseline.

TrojanNN có thêm chi phí clean-model-informed trigger optimization; ngân sách
PGD/patch của nó được ghi riêng, không gọi là equal-L2/Linf với các operator khác.
TEST chỉ dùng evaluation sau khi khóa trigger, không chọn trigger theo điểm TEST.

Preset cố định theo BackdoorBench commit
`f02e3534645f0ee63d6848653062cd6c0d6c400d`: chọn **hai input units** có tổng
trọng số outgoing tuyệt đối cao nhất của final Linear `regression.7`; target
activation100, tối đa1000 bước sign-PGD, alpha0.1, radius0.3, tolerance0.001.
CSI adaptation dùng patch3×3 cuối trục frequency/packet trên cả ba antenna,
khởi tạo patch trắng; input ngoài patch bằng0 khi tối ưu. Injection là
`clip(x + dose * patch, 0, 1)` trên float32, không normalize lại.
**Radius0.3 quanh patch trắng không phải cap0.3 trên residual**: giá trị patch
nằm trong[0.7,1]. Lưu loss/neuron/pattern/source hashes; không tự retune layer
hoặc activation target nếu loss/hiệu quả thấp. Không có bảo đảm thắng A1.

Checkpoint Clean lịch sử không có input-identity hash tại thời điểm train.
Runner kiểm tra recipe/fingerprint/state/SHA và audit toàn bộ dữ liệu hiện tại;
điều đó không chứng minh ngược lại TRAIN bytes lịch sử giống hoàn toàn.

## Lệnh trên PowerShell máy Vân

Nếu repo nằm ở `C:\Users\Van\code`, chạy cả block này. Nếu repo ở chỗ khác,
chỉ sửa `$Repo` cho đúng. Không pull vào thư mục có thay đổi chưa lưu mà không
kiểm tra; `--ff-only` sẽ dừng nếu không fast-forward được. Không cập nhật source
checkout đang được một job sử dụng; chờ job đó xong hoặc dùng checkout riêng.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code'
    git -C $Repo pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Pull loi; dung.' }
    $S = Join-Path $Repo 'remote_tools\Start-MMFiP1S1TrojanNN.ps1'
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
    if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua Launch.' }
}
```

`Prepare` kiểm tra dữ liệu/clean reference, xây trigger và smoke batch32 trên
CUDA; **chưa train full victim**. Xem output/receipt rồi mới chạy:

```powershell
$S = 'C:\Users\Van\code\remote_tools\Start-MMFiP1S1TrojanNN.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Nếu dùng Python hoặc clean checkpoint ở vị trí khác, dùng **cùng bộ tham số**
cho Prepare/Launch/Status/Export, ví dụ:

```powershell
$S = 'C:\Users\Van\code\remote_tools\Start-MMFiP1S1TrojanNN.ps1'
$Opt = @{
    DataHome = 'C:\Users\Van\backdooranalog'
    PythonExe = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
    CleanCheckpoint = 'C:\Users\Van\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1\seed42\clean\checkpoint.pt'
    Outdir = 'C:\Users\Van\backdooranalog\runs\mmfi_p1s1_hpeli_trojannn_s42_v1'
}
& $S -Mode Status @Opt
```

`Launch` kiểm tra lại gate trước khi chạy ẩn; source/input/trigger/runtime không
khớp thì dừng, không tự hạ batch, chuyển AMP, retry hay sửa checkpoint. Resume
chỉ dùng output và hợp đồng đã chuẩn bị tương ứng. Không chạy hai helper cùng lúc.

## Xem tiến trình và xuất bảng

```powershell
$S = 'C:\Users\Van\code\remote_tools\Start-MMFiP1S1TrojanNN.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = 'C:\Users\Van\backdooranalog\runs\mmfi_p1s1_hpeli_trojannn_s42_v1'
Get-ChildItem -LiteralPath "$R\logs" -File -Filter '*.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 12 -Encoding UTF8 }
```

Khi evaluate hoàn tất:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Export
```

Giữ T-MPJPE và Target-PCK tại dose1, Clean MPJPE/PA-MPJPE và DCE trung bình năm
dose dương; lưu diagnostics/no-trigger và provenance cùng output. Không chọn
neuron/layer/seed/target/hyperparameter theo official TEST hay đổi số lịch sử.
Seed42 đơn lẻ không cung cấp độ lệch chuẩn giữa seed. Kiểm thử code/CPU local
không thay cho audit dữ liệu thật, CUDA/VRAM và hiệu quả thực nghiệm trên máy Vân.

Verification local 2026-10-29: **124 test mới +39 regression passed**, gồm
operator/HPE-Li CPU, checkpoint/config/receipt guards, Windows PowerShell parse
và quoting, resume toy khớp tuyệt đối weights/momentum/RNG. Checkpoint Clean
P1S1 lịch sử thật đọc/fingerprint/strict-state đạt; hai bước tối ưu CPU dùng
checkpoint đó đạt finite. Không dùng các bước smoke này làm trigger final hay
kết quả TEST; Prepare trên máy chạy sẽ tạo trigger1000-step theo preset đã khóa.

Nguồn thuật toán: [BackdoorBench TrojanNN](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/attack/trojannn.py),
[TrojanNN, NDSS 2018](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-5_Liu_paper.pdf).
Chi tiết CSI adaptation và source pin được runner ghi trong manifest; không
gộp với FTrojan-CSI (hai thuật toán khác nhau).
