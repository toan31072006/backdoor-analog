# Adap-Patch / HPE-Li / MM-Fi P1S1 / seed42 — Vân4090

Material Passport: `mmfi_p1s1_hpeli_adap_patch_s42_4090_v1`, origin date
2026-11-02 Asia/Bangkok, ARS experiment-agent. AI hỗ trợ triển khai/test; đây là
**T1 exploratory additional baseline**, chưa phải bảng nhiều method/seed đầy đủ.
Local tests không thay cho provenance dữ liệu và CUDA smoke thật trên Vân.
Theo yêu cầu sau đó của người dùng, **không audit/hash lại dataset**: tái dùng
audit P3S3 đã PASS, kiểm tra subset P1S1 và file-stat (size/mtime), không đọc lại
nội dung các file CSI/pose. Thiếu audit gốc hoặc file thay đổi thì dừng, không
tự bắt đầu một full re-audit tốn thời gian.

## Không sửa run P3S3 đang chạy

P1S1 dùng **checkout mới** `C:\Users\Van\code_adap_patch_p1s1` và output riêng
`C:\Users\Van\backdooranalog\runs\mmfi_p1s1_hpeli_adap_patch_s42_4090_v1`.
Không pull/sửa checkout `code_adap_patch_p3s3`, manifest, gate hoặc checkpoint
P3S3 đang chạy; không kill/pause process khác. Checkpoint P3S3 không warm-start
P1S1. Không train lại Clean hay A1, không sửa kết quả lịch sử.

Mặc định P1S1 chạy exclusive. Nếu dùng **`-AllowConcurrentP3S3`**, chỉ cho phép
**1 P1S1 + 1 Adap-Patch P3S3** gốc chạy cùng, gồm proxy chain bình thường của venv
Windows. Không phải hai P1S1 hoặc bất kỳ hai Python jobs nào. Worker P3S3 phải có
đúng script/output/protocol/seed/venv/source pin; unknown processes/job khác vẫn
bị chặn. Controller/probe chỉ CPU hoặc sống ngắn; không giữ CUDA context lúc
spawn worker. Guard là snapshot chống chạy nhầm, không phải OS GPU reservation.

Chế độ sharing lưu trong plan, phải dùng **cùng switch** ở Prepare/CheckOnly/
Launch/resume. Nếu P3S3 kết thúc, vẫn dùng switch cũ cho output P1S1 đã khóa;
không sửa manifest để đổi policy. Không cần có peer đang sống khi resume.
Nếu cần khởi chạy lại P3S3 trong lúc P1S1 đang chạy, launcher P3S3 gốc vẫn
exclusive và có thể từ chối: không bypass nó hoặc sửa source đang khóa.

**Giữ SSH mở**, Launch là foreground, không phải tmux/Task Scheduler và không
cam kết sống sau disconnect. Một snapshot GPU34% không chứng minh máy rảnh hay
hai job nhanh hơn. Có thể tranh CPU/đĩa/VRAM; không cam kết speedup hoặc vừa bộ
nhớ. Khi lỗi OOM/nonfinite: dừng và gửi log, không tự retry/shrink batch/AMP.

## Recipe / source adaptation

- Full **P1S1: 133056 TRAIN / 33264 TEST**, protocol1-s1, split seed0. S1 là
  subject draw **riêng từng action**: 448/112 sequences, disjoint sequences,
  có thể chung subject qua action. Không gọi là held-out global subject/scene.
- HPE-Li scratch, **50 epochs/final49**, batch32, workers0, drop_last=False,
  ordinary MPJPE, SGD lr=.001 / momentum=.9 / weight decay0, FP32; không đổi
  recipe P3S3 ngoài split, không chọn best TEST hoặc thêm surrogate/generator.
- Source operator giữ nguyên file `attack/adap_patch_mmfi.py` đã publish với
  P3S3: repo tác giả *Circumventing-Backdoor-Defenses*, ICLR2023, pin
  `e7c070344a5c6004d6fbcdf0720e2d60ec808953`. **GTSRB trigger preset**, không phải
  GTSRB dataset. K4 TRAIN alphas [.5,.2,.5,.3]; TEST firefox rồi trojan alpha1.
  Preset CIFAR thiếu `phoenix_corner2_32.png`, không tự thay thế.
- RGB3 map sang antenna3; mark32x32 bilinear và mask nearest sang114x10; giữ
  normalized float CSI, không renormalize/quantize PNG hoặc comparison cap.
  Đây là digital CSI/pose/dose adaptation, không image reproduction/RF OTA.
- Common seed42 poison10%: **13305 poison**, paired fixed U(.2,1), pivot1,
  target joints[2,3], Z40*d. Clean-label cover20%: **26611 cover**, disjoint,
  giữ GT. Tổng **39916 inputs bị sửa**, xấp xỉ30%, khác A1 chỉ10% poison.
  Giữ cover:poison 2:1 theo requested fractions (floor ở từng nhóm); không
  claim equal-modified-data/equal-noise budget, không bỏ cover để làm yếu method.
- Sáu dose0,.2,.4,.6,.8,1; T-MPJPE/Target-PCK tại d1 trên[2,3], scale clean GT
  [5,12], threshold.2; Clean MPJPE/PA17 khớp; DCE mean năm dose dương, không
  thêm dose0 vào mean. Target-PCK không phải clean PCK hoặc pose ASR.
- Lưu all predictions/truth/targets/ordered IDs và same-victim no-trigger
  controls. Run đơn lẻ không có independent clean-trained-model+trigger control
  và không tự ghép/sửa bảng T1 cũ. Không đổi preset/threshold theo TEST.

## 1. Clone checkout P1S1 riêng

```powershell
$Repo = 'C:\Users\Van\code_adap_patch_p1s1'
if (Test-Path -LiteralPath $Repo) { throw 'Checkout da ton tai; gui output, khong ghi de.' }
git clone https://github.com/toan31072006/backdoor-analog.git $Repo
if ($LASTEXITCODE -ne 0) { throw 'Clone loi; dung.' }
$S = Join-Path $Repo 'remote_tools\Start-MMFiP1S1AdapPatch4090.ps1'
```

Nếu checkout P1S1 đã tồn tại và muốn update sau này, kiểm `git status` sạch rồi
mới pull --ff-only, không reset/xóa file. Không pull P3S3 đang train.

## 2. Prepare cùng P3S3 (không train đầy đủ)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect -AllowConcurrentP3S3
if ($LASTEXITCODE -ne 0) { throw 'Inspect loi; dung.' }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -AllowConcurrentP3S3
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua launch.' }
```

Prepare: reuse receipt P3S3 + P1S1 identity/subset/file-stat checks, **không hash
lại CSI/pose**; cộng CUDA physical B32 với4 poison/4 cover/
24 clean (mọi trigger group), hai SGD steps. Assets runtime download pinned
hash, không phân phối ảnh upstream. Đây là shared-context smoke, chưa train đủ
50 epochs, không guarantee efficacy hoặc future VRAM. Không tự cài/đổi env;
dùng env `dose-backdoor` hiện có hoặc truyền `-PythonExe` nhất quán.

Guard nhận P3S3 gốc chỉ đọc từ `code_adap_patch_p3s3`, không chấp nhận chỉ đưa PID
hay tên python để bỏ qua. Nếu peer khác đường dẫn/source hoặc có job thứ ba,
gửi CIM CommandLine để kiểm tra; không tắt guard. Nếu không muốn sharing, dùng
output mới/chế độ exclusive trước Prepare; không sửa output đã bind policy.

## 3. Sau Prepare đạt, Launch riêng trong SSH thứ hai

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -AllowConcurrentP3S3
```

Giữ **cả SSH P3S3 và SSH P1S1** mở. Checkpoint atomic mỗi epoch kèm optimizer/
RNG, source/runtime/input/asset/poison/cover bindings; resume cùng lệnh nếu đã
kiểm tra worker trước đó thật sự dừng. Full TEST six-dose sau final49.

## 4. Status chỉ đọc

```powershell
$S = 'C:\Users\Van\code_adap_patch_p1s1\remote_tools\Start-MMFiP1S1AdapPatch4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = 'C:\Users\Van\backdooranalog\runs\mmfi_p1s1_hpeli_adap_patch_s42_4090_v1'
Get-Content -LiteralPath "$R\adap_patch\console.log" -Tail 12 -Encoding UTF8
```

Kết quả cuối `$R\comparison.md`, `$R\adap_patch\evaluation.json`,
`per_sample.npz`, `checkpoint.pt`. Không suy completion từ GPU0%, PID hay file
tồn tại. Status metadata báo UNKNOWN về liveness nếu không thể xác nhận.

## Kiểm thử trước publish — 2026-11-02

**256 tests passed**, exit0: 73 kiểm tra P1S1/peer-sharing/launcher mới và183
kiểm tra hồi quy Adap-Patch/P3 cũ, gồm model HPE-Li CPU B32 tổng hợp. 19 cảnh
báo pin_memory là do môi trường CPU. Có kiểm tra exact optimizer/RNG resume,
native trigger/cover labels, Windows venv ancestry, hai worker P3 bị chặn,
pending job thứ ba bị chặn, sharing flag chuyển đúng mọi phase, full six-dose
metric/NPZ, và reuse P3 audit không đọc/hash file CSI gốc. DryRun thực tế không
tạo output/network/GPU; P3S3 source pin vẫn nguyên byte. Không có kiểm thử CUDA
hai job trên Vân hoặc kết quả attack mới từ các unit tests này.
