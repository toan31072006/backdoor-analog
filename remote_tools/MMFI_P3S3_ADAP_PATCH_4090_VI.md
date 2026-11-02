# Adap-Patch adapted — HPE-Li / MM-Fi P3S3 / seed42 / Vân4090

## Phạm vi và trạng thái

- Material Passport: `mmfi_p3s3_hpeli_adap_patch_s42_4090_v1`, origin date
  **2026-11-02 Asia/Bangkok**, ARS experiment-agent; AI hỗ trợ triển khai/kiểm tra.
- Đây là baseline bổ sung exploratory, một seed, thuộc câu hỏi T6 về split.
  Không thay bảng T1/P1S1, không phải toàn bộ bảng nhiều method/seed đã hoàn tất.
- Chỉ một victim Adap-Patch mới. Không train lại Clean hoặc A1, không chép hoặc
  đổi nhãn artifact lịch sử. Chưa có kết quả full MM-Fi/CUDA trên máy Vân chỉ vì
  code và unit tests đã sẵn sàng. Local tests không thay cho audit dữ liệu thật.
- Tối đa **1 job GPU**, `cuda:0`, RTX4090. Không cho chạy ghép bằng switch bỏ guard.
  Launcher không kill/pause process khác, không tự retry, giảm batch hoặc đổi env.
- **Giữ SSH mở** khi Launch. Foreground không phải tmux, không có cam kết sống
  qua disconnect/reboot. Nếu bị ngắt, kiểm tra Status/process trước khi gọi lại;
  chỉ checkpoint atomic của epoch hoàn tất mới được resume.

### Kiểm tra local trước publish — 2026-11-02

- **183 tests passed**, exit0 trong một lượt gộp: 95 tests Adap-Patch/launcher/
  real-model smoke và 88 hồi quy P3 hiện có. 10 cảnh báo pin-memory là do máy
  kiểm thử CPU không có accelerator, không phải kết quả CUDA.
- HPE-Li thật, synthetic physical B32, hai bước SGD: loss/gradient hữu hạn,
  weights thay đổi, output đúng `[32,1,17,3]`; poison/cover/clean và dose0 identity
  được kiểm tra. Toy CPU kiểm tra optimizer/momentum/RNG exact resume.
- Năm PNG tác giả được tải và đối chiếu SHA/geometry thực tế. Bốn masks không
  biến mất sau resize: supports `[17,159,21,87]` trên1140 vị trí; không claim
  mọi trigger có diện tích bằng nhau.
- DryRun thực tế không tạo output/tải assets/dùng GPU; PowerShell parse và
  phase forwarding dùng mock. Independent review không thấy lỗi chặn.
- **Giới hạn:** chưa chạy audit full dữ liệu máy Vân, smoke CUDA, full victim
  training hoặc full TEST; không có efficacy, runtime GPU hay đảm bảo hội tụ.

## Nguồn và adaptation đã khóa trước khi chạy

Paper: *Revisiting the Assumption of Latent Separability for Backdoor Defenses*,
ICLR 2023. [Repo tác giả](https://github.com/Unispac/Circumventing-Backdoor-Defenses),
commit `e7c070344a5c6004d6fbcdf0720e2d60ec808953`.
Đối chiếu `poison_tool_box/adaptive_patch.py`, `config.py` và README, không gọi đây
là reproduction nguyên bản của classification.

- Giữ nguyên **preset trigger GTSRB** trong config tác giả: TRAIN có bốn trigger
  riêng lẻ, alphas `[0.5, 0.2, 0.5, 0.3]`; TEST áp tuần tự `firefox_corner_32.png`
  rồi `trojan_square_32.png`, alphas `[1,1]`. Đây chỉ là preset **trigger**;
  dataset thí nghiệm vẫn là MM-Fi P3S3, không phải GTSRB.
- Không dùng preset CIFAR thiếu file `phoenix_corner2_32.png` trong pinned tree;
  không tự tạo file thay thế hoặc đổi test trigger sau khi nhìn TEST.
- Assets lấy từ URL tác giả tại commit pin, kiểm SHA256/shape/range, giữ cache
  bất biến. Không phân phối lại nguyên code/ảnh upstream chưa xác minh license.
- RGB ba kênh được map số học sang ba antenna; ảnh 32x32 map sang mặt phẳng
  subcarrier114 x packet10. Đây là **digital tensor adaptation**, không phải
  phép biến đổi RF có ý nghĩa vật lý hoặc bằng chứng stealthiness over-the-air.
  Mark nội suy bilinear, mask nearest, giữ float-domain normalized `[0,1]`, không
  chuẩn hóa lại/quantize thành PNG sau injection. Không thêm comparison peak cap.
- Poison identities và paired TRAIN doses `U(.2,1)` dùng kế hoạch chung seed42.
  Giữ nhiều trigger và disjoint clean-label cover. Chia nhóm theo thứ tự **sorted
  rank** của poison/covers như nguồn, không chọn ngẫu nhiên trigger từng epoch.
- TRAIN alpha bằng alpha gốc nhân dose; TEST áp hai alpha gốc nhân dose. Payload
  classification chuyển thành pose target chung. DCE là kiểm tra khả năng điều
  khiển dose của **adaptation**, không phải objective được paper gốc cam kết.

### Ngân sách phải báo riêng

Ví dụ Adap-Patch trong README tác giả dùng poison `.003` và cover `.006`.
Profile này giữ **tỷ lệ cover:poison 2:1**, nhưng dùng common poison `.1`:

- TRAIN 240570: **24057 poison (10%)** + **48114 cover (20%)**, disjoint.
- Tổng số input TRAIN bị sửa là **72171 (30%)**, không phải 10% như A1 không cover.
- Poison đổi pose target; cover có trigger nhưng **giữ nguyên pose thật**.
- Không claim equal-modified-data/equal-noise budget. Cùng recipe không làm L2,
  Linf hoặc diện tích trigger bằng nhau. Không bỏ covers để làm ngang ngân sách
  rồi vẫn gọi nguyên Adap-Patch. Muốn matched-budget cần study riêng được duyệt.

## Recipe / metrics

Giữ recipe HPE-Li P3S3 của `run_mmfi_p3_splits._configuration`:
seed42, full official TRAIN **240570** / TEST **80190**, split seed0,
**50 epochs/final49**, physical batch32, workers0, SGD **lr=.001, momentum=.9,
weight decay=0**, ordinary MPJPE, FP32, drop_last=False.
Resolved plan phải khớp các giá trị này, không tự lấy recipe
MetaFi/PiW hoặc best TEST checkpoint. Không surrogate, learned generator hoặc
trigger-optimization stage; chưa đo runtime nên không hứa ETA ngang A1.

Payload pivot1, target joints `[2,3]`, xoay Z tối đa40 độ nhân dose. TEST doses
`0,.2,.4,.6,.8,1`. Báo T-MPJPE / Target-PCK tại dose1; PCK threshold `.2`, scale
từ clean GT `[5,12]`; Clean MPJPE / PA-MPJPE trên17 khớp; DCE trung bình **năm
dose dương**. Target-PCK không phải clean PCK hoặc pose-level ASR.

Giữ same-victim no-trigger controls và predictions/truth/targets/ordered IDs để
audit. Run này không có independent clean-trained-model + trigger control;
không trình bày hàng attack đơn lẻ như full controlled comparison. Không tune
target/threshold/preset bằng TEST hoặc loại method vì kết quả bất lợi.

## 1. Lấy checkout riêng trên PowerShell máy Vân

Không pull/sửa checkout của job đang chạy. Dùng thư mục riêng sau đây; nếu thư
mục đã tồn tại và bẩn, dừng để kiểm tra, không reset hoặc ghi đè.

```powershell
$Repo = 'C:\Users\Van\code_adap_patch_p3s3'
if (!(Test-Path -LiteralPath $Repo)) {
    git clone https://github.com/toan31072006/backdoor-analog.git $Repo
    if ($LASTEXITCODE -ne 0) { throw 'Clone loi; dung.' }
} else {
    $Dirty = git -C $Repo status --porcelain
    if ($LASTEXITCODE -ne 0 -or $Dirty) { throw 'Checkout khong sach; dung, khong reset.' }
    git -C $Repo pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Pull loi; dung.' }
}
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3S3AdapPatch4090.ps1'
```

Không cần zip checkpoint hoặc cài lại env tự động. Mặc định helper tìm existing
`C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`, rồi các đường dẫn
Scripts/mmfi-p3; dùng `-PythonExe` nếu cần. Cần torch, numpy, scipy, PyYAML,
Pillow và các phụ thuộc HPE-Li hiện có.

## 2. Inspect / DryRun / Prepare

Inspect/DryRun không train, không tải assets hoặc dùng GPU. Prepare cần GPU
không có compute job khác: full input audit + tải/check assets + real CUDA B32
smoke có poison/cover/clean, **không train đủ50 epoch**.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
if ($LASTEXITCODE -ne 0) { throw 'Inspect loi; dung.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode DryRun
if ($LASTEXITCODE -ne 0) { throw 'DryRun loi; dung.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua launch.' }
```

Gửi output Prepare để xem trước. Nếu thiếu data, sai checksum, source/runtime
binding, GPU busy hoặc smoke lỗi: dừng, không bỏ guard hay tự sửa manifest.

## 3. Launch riêng sau khi Prepare đạt

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
if ($LASTEXITCODE -ne 0) { throw 'Launch loi; gui log, khong tu retry.' }
```

Giữ SSH mở. Cùng output + source/env/config/input không đổi sẽ resume checkpoint
epoch đã hoàn tất; không warm-start victim từ A1/Clean. Full evaluation theo sau
final49, không chấm checkpoint đang ghi. File tồn tại/GPU0% không chứng minh xong.

## 4. Xem tiến độ / kết quả

Mở một SSH khác, đặt lại `$Repo`/`$S` nếu biến chưa có:

```powershell
$S = 'C:\Users\Van\code_adap_patch_p3s3\remote_tools\Start-MMFiP3S3AdapPatch4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$Run = 'C:\Users\Van\backdooranalog\runs\mmfi_p3s3_hpeli_adap_patch_s42_4090_v1'
Get-ChildItem -LiteralPath $Run -Recurse -File -Filter '*.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 12 -Encoding UTF8 }
```

Kết quả `evaluation.json`, `per_sample.npz`, `comparison.md`; checkpoint cuối
được giữ. Source/plan/assets/input/poison/cover/checkpoint/prediction hashes ràng
buộc cùng run. Chỉ báo complete khi final49 và full TEST artifacts qua audit.
