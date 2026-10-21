# MM-Fi: full T1/T2/T3 trên MetaFi++ và GraphPose-Fi

## Phạm vi đã khóa

Đây là batch **cross-backbone trên cùng MM-Fi**, không phải dataset thứ hai,
không phải chạy full DT-Pose và không bao gồm defense. Kết quả HPE-Li, các
checkpoint và manifest lịch sử giữ nguyên.

- Hai backbone riêng: `metafiplusplus` và `graphposefi`.
- Seeds 42/43/44, mỗi backbone có 13 victim/seed: Clean, A1, sáu baseline
  BadNets/WaNet/FTrojan/FIBA/SIG/BppAttack, A2/A3/A4, A1 rho=.2/.4.
- Tổng **78 victim trainings mới**. A1 rho=.1 trong cùng backbone/seed được
  dùng chung cho T1/T2/T3, không train lặp theo dose.
- Full MM-Fi protocol1-s1, random split .8, split seed 0; TRAIN 133056, TEST
  33264. Giữ nguyên input identities, trigger, poison/dose sampler và target
  của bộ ba bảng HPE-Li. Không thêm Blended, không đổi carrier để ưu ái backbone.
- Cùng 50 epochs; dùng **checkpoint cuối epoch 49**, không chọn best TEST.
  Khác biệt này với code tác giả được khai báo, không gọi là reproduction
  nguyên xi các số clean trong bài gốc.
- Scratch `pretrained=False`: không tự tải ImageNet weights hoặc MAE encoder.
  MetaFi++ là backbone từ nhánh baseline của DT-Pose, không phải pipeline
  pretraining + decoding của DT-Pose.
- Constructor MetaFi++ công khai mặc định ImageNet: initializer reset Conv và
  BN affine nhưng vẫn giữ BN running statistics từ ImageNet. Chế độ scratch
  của batch này dùng statistics identity; đây là khác biệt initialization được
  công bố, không gọi là chạy đúng từng bit code/clean score tác giả.

| Backbone | Recipe cố định |
| --- | --- |
| MetaFi++ | SGD lr=.01, momentum=.9, weight decay 0, batch 32, không scheduler |
| GraphPose-Fi | AdamW lr=.0003, **physical/load batch 32, accumulation 8, effective batch 256**, weight decay .02 chỉ nhóm decay, norm/bias/1D không decay; warmup 5 epoch + cosine; gradient clipping norm 1 sau accumulation |

Recipe tham chiếu nhánh MM-Fi trong code tác giả DT-Pose `train_pose.py` và
GraphPose-Fi `train.py`/`utils.py`. Không dùng recipe HPE-Li lr=.001 cho mọi
backbone. GraphPose-Fi source mặc định physical/effective batch 256; batch này
khóa **physical 32 + effective 256** trước khi chạy để phù hợp hai job RTX4090.
Không tự giảm batch sau OOM. Author GraphPose-Fi dùng
parameter sets cho nhóm decay; adapter giữ nguyên membership nhưng dùng thứ
tự tham số ổn định để checkpoint optimizer có thể resume.

**Khác biệt training cần công bố:** BatchNorm dùng statistics của microbatch32,
không phải statistics của cả batch256; accumulation không biến hai cách này
thành tương đương. Mỗi nhóm accumulation dùng MPJPE theo số sample thật; nhóm
cuối epoch nếu chưa đủ 256 vẫn được chuẩn hóa theo số sample thật, optimizer
step rồi reset. Không sao chép global-step counter xuyên epoch hoặc cách
underweight nhóm cuối dở trong vòng lặp nguồn. Vì vậy đây là **memory-adapted
recipe**, không phải original-paper reproduction chính xác hay cải tiến method.

Các guard cũ trong `models/factory.py` **không phải kết luận về code tác giả**:
MetaFi++ còn BatchNorm sau decoder nên output cuối không bị buộc không âm;
graph 17 khớp của GraphPose-Fi tương thích target MM-Fi hiện tại. Sidecar mới
không sửa factory/trainer/evaluator HPE-Li đã được hash trong các kết quả cũ.

## Pilot hiện hành 2026-10-21: chỉ A1 seed 42, tối đa hai job

Người dùng chốt **chưa thêm IAD/LIRA, chưa chạy dataset thứ hai**. HPE-Li
T1--T3 và defense đã có kết quả thì không train lại. Bước chạy được ngay là
A1 trên MetaFi++ và GraphPose-Fi, seed42, full50epochs/rho=.1/sáu dose, với
clean control cùng backbone/seed. Đây chưa phải full ma trận hoặc bảng ba seed.

Chạy trong **PowerShell trên máy Vân** sau khi controller/worker cũ đã dừng:

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
cd "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull lỗi" }
$CB = "$env:USERPROFILE\backdooranalog\runs\mmfi_cross_backbones_s42_43_44_v1"

& .\remote_tools\Start-A1CrossBackboneTask.ps1 `
  -Outdir "$CB" -JobsPerGpu 2 -GraphLoadBatchSize 32 -PilotA1Seed42 -DryRun
if ($LASTEXITCODE -ne 0) { throw "Pilot preflight lỗi; chưa chạy." }

& .\remote_tools\Start-A1CrossBackboneTask.ps1 `
  -Outdir "$CB" -JobsPerGpu 2 -GraphLoadBatchSize 32 -PilotA1Seed42
```

Dry-run phải báo **78 planned / 4 selected / 2 slots**: hai A1 + hai clean
dependencies, không phải train mới cả bốn. Clean hợp lệ được audit rồi skip;
clean dở resume epoch checkpoint. Nếu GraphPose-Fi clean42 chưa xong, scheduler
phải hoàn tất clean trước Graph A1; không hứa hai A1 bắt đầu đồng thời.
Clean43/44, các baseline và ablation khác **không được chọn** trong lượt này.

Helper không đổi Python scientific sources, source hash, recipe hoặc manifest.
GPU0% chưa chứng minh không có CPU controller. Ngoài kiểm tra task/process,
helper thử lấy rồi nhả khóa byte0 của `run.lock` hiện có trước khi đăng ký task;
không tạo/xóa/ghi đè file khóa. Controller vẫn lấy khóa thật khi chạy để xử lý
race. Nếu báo khóa không khả dụng, kiểm tra owner hoặc quyền truy cập; **không
xóa lock, đổi output để né owner hoặc kill toàn bộ Python**.

```powershell
foreach ($kind in @("stdout","stderr","wrapper-error")) {
  Get-ChildItem "$CB.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Encoding UTF8 -Tail 15 }
}
foreach ($model in @("metafiplusplus","graphposefi")) {
  $log = Join-Path $CB "$model\seed42\a1\console.log"
  if (Test-Path -LiteralPath $log) {
    $log
    Get-Content -LiteralPath $log -Encoding UTF8 -Tail 10
  } else {
    Write-Host "$model A1 chưa có log: xem controller và clean42 trước."
  }
}
```

Kết thúc subset, dòng `selected cells finished; FULL TABLES INCOMPLETE` là
đúng phạm vi pilot, không tự là lỗi. Cần đủ evaluation A1 và clean42 của cả hai
backbone được audit, số hữu hạn, checkpoint đúng49 và log không lỗi; một dòng
task `Ready` hoặc số loss giảm chưa phải kết quả attack đã hoàn tất.

**Bước defense đã chốt để làm tiếp:** NAD trên MetaFi++ và ANP trên
GraphPose-Fi, seed42, ghép No-defense của chính A1 tương ứng; giữ metrics
target/clean/triggered-GT/cùng-khớp, PA và sáu-dose DCE diagnostics. Tại commit
launcher pilot này, adapter/runner defense cho hai backbone **chưa được triển
khai/kiểm chứng**. NAD hiện hardcode layer HPE-Li; không trỏ runner HPE-Li vào
checkpoint MetaFi++/GraphPose-Fi rồi gọi là defense cross-backbone. Phải audit
hook/layer coverage và khóa memory-adapted recipe riêng trước GPU pilot. Không
chạy lại defense HPE-Li để thay cho hai nhánh đó. Chưa có lệnh hay kết quả mới
IAD/LIRA và không đổi tên các learned-carrier drafts thành reproduction.

Kiểm thử bổ sung: **47 test launcher/runner đạt, exit0**, gồm native Windows
byte-lock đang được giữ vẫn chặn task trước registration và giữ nguyên bytes.
**2 test defense preflight đạt, exit0** trên model thật/dữ liệu tổng hợp: NAD
MetaFi++ có gradient qua fused features (`bn2`, `tf`, `decode.2`), teacher không
bị cập nhật; ANP GraphPose-Fi thay/mask ordinary BN và giữ nguyên source model.
LayerNorm attention/graph không được ANP BN mask cover. Đây chỉ là porting
probes, chưa khóa repair recipe hoặc xác nhận resume/end-to-end MM-Fi defense.

## Trước khi chạy đầy đủ

Code và test synthetic/local không xác nhận clean utility hay full GPU run.
Chạy **clean pilot trước**, kiểm tra loss hữu hạn, output đủ 17 khớp/signed,
clean MPJPE/PA-MPJPE và nhật ký training hợp lý. Gate kỹ thuật chỉ bắt lỗi
shape/NaN/checkpoint, không tự chứng minh reproduce clean accuracy.

Không tune recipe, target, epsilon hay chọn seed từ TEST. Nếu clean model chưa
hợp lý, dừng batch, audit data/preprocessing/kiến trúc và khóa **plan/output mới**
trước khi train attack; không chạy đến khi có số target đẹp. Giữ lại pilot đã
chạy. Clean pilot và full dùng cùng manifest nên clean hợp lệ sẽ được audit rồi
skip, không train lại.

Theo yêu cầu mới nhất của người dùng, mặc định **2 job/GPU**, Windows workers=0.
Một synthetic CPU training-forward GraphPose-Fi batch1 giữ khoảng **99.35 MiB**
autograd activation storage (đã loại parameter/buffer storage); scale tuyến tính
đến physical256 là khoảng **24.84 GiB/job**, chưa gồm optimizer/workspace. Đây
là **ước lượng bộ nhớ, không phải đo CUDA peak**; không dùng default256 rồi hứa
rằng RTX4090 chạy nổi hai job. Default physical32 giảm phần activation theo batch,
nhưng vẫn phải đo VRAM thực tế. Nếu OOM, dừng owner và chủ động đổi runtime về
1 slot; không tự thay frozen recipe hoặc bỏ kiểm tra để che lỗi.

## Lệnh trên máy Vân — PowerShell

Không dán prompt `(base) PS ...>` vào terminal. Các lệnh bên dưới dành cho
Windows PowerShell, không phải CMD hoặc Bash.

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
cd "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull lỗi" }
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$CB = "$env:USERPROFILE\backdooranalog\runs\mmfi_cross_backbones_s42_43_44_v1"

python -u .\ATKBackd\run_a1_cross_backbones.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$CB" `
  --backbones metafiplusplus graphposefi --devices cuda:0 `
  --jobs-per-gpu 2 --num-workers 0 --graph-load-batch-size 32 --dry-run
```

Hai model là implementation độc lập theo source đã pin và dùng torchvision
ResNet34; không cần clone thêm repo tác giả hoặc tải model weights. Source
passport ghi commit/hash, khác biệt interface/initialization và phạm vi test.
Giữ nguyên phiên bản Python/torch/torchvision/NumPy/SciPy/PyYAML sau khi tạo
manifest; nâng package là runtime khác, không bỏ source/runtime check để resume.

Dry-run chỉ resolve/kiểm tra plan: không đọc raw data/checkpoint, không dùng
GPU, không train và không tạo output. Thiếu data vẫn có thể dry-run thành công;
đó không phải preflight thực tế đã đạt.

Chạy clean pilot bằng task độc lập để đóng VS Code/ngắt SSH không làm mất job:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1CrossBackboneTask.ps1 `
  -Outdir "$CB" -JobsPerGpu 2 -GraphLoadBatchSize 32 -PilotClean
```

Sau khi cả sáu clean model xong, đọc/audit kết quả và kiểm tra utility. Nếu đã
hợp lý, chạy full **cùng output** (bỏ `-PilotClean`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1CrossBackboneTask.ps1 `
  -Outdir "$CB" -JobsPerGpu 2 -GraphLoadBatchSize 32
```

Muốn đổi số slot: thay `-JobsPerGpu 2` bằng `-JobsPerGpu 1` khi controller cũ
**đã dừng**. Helper từ chối output đang active, không tự kill
owner/YOLO/defense và không xóa lock/checkpoint. Task vẫn cần máy bật, không
sleep/shutdown; S4U không bảo đảm truy cập ổ mạng hoặc file EFS.

`-GraphLoadBatchSize` nhận 32/64/128/256, effective batch vẫn 256; CLI tương ứng
`--graph-load-batch-size`. Đây là **scientific setting được hash**, không phải
runtime option đổi tùy lúc. Chọn giá trị khác phải dùng plan/output mới; không
đổi giữa clean pilot và full. Chỉ dùng 256 trên phần cứng đủ VRAM sau khi đo,
không tự bật vì còn vài GB trống. Việc đổi số job (1/2) không đổi recipe.

## Theo dõi và resume

```powershell
Get-ScheduledTask -TaskName MMFI-A1-CrossBackbones |
  Select-Object TaskName,State

foreach ($kind in @("stdout","stderr","wrapper-error")) {
  Get-ChildItem "$CB.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    ForEach-Object {
      $_.FullName
      Get-Content $_.FullName -Encoding UTF8 -Tail 20
    }
}

Get-ChildItem "$CB\*\seed*\*\console.log" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 2 |
  ForEach-Object {
    $_.FullName
    Get-Content $_.FullName -Encoding UTF8 -Tail 12
  }

nvidia-smi
```

`Running` chỉ là trạng thái task, không chứng minh training tiến triển.
Epoch/iteration/log timestamp mới và GPU activity mới giúp xác nhận tiến trình.
`Ready` có thể là thành công hoặc lỗi: luôn đọc stdout/stderr.

Nếu máy/process bị dừng, bảo đảm task/worker cũ không còn rồi gọi lại đúng lệnh
full ở trên, giữ nguyên `$CB`. Cell hoàn tất chỉ skip sau artifact audit; cell
dở resume atomic checkpoint gần nhất, có thể mất phần epoch chưa được lưu.
Không xóa cache/manifest để ép chạy tiếp khi config/source mismatch.

CLI trực tiếp cho một cell hoặc audit (không khởi chạy thêm nếu task đang chạy):

```powershell
python -u .\ATKBackd\run_a1_cross_backbones.py --outdir "$CB" --status
python -u .\ATKBackd\run_a1_cross_backbones.py --outdir "$CB" --check-only

python -u .\ATKBackd\run_a1_cross_backbones.py --outdir "$CB" `
  --devices cuda:0 --jobs-per-gpu 2 `
  --cells graphposefi__s44__a1
```

`--status` đếm artifact, không phải verification. `--check-only` chỉ audit
artifact hiện có, không train/inference. `--export-only` audit rồi xuất đủ ba
bảng từng backbone khi đủ 39 cell/backbone; thiếu seed phải báo partial/missing,
không dùng số frame để thay thế training seed.

## Output và metric

Checkpoint/log: `$CB\<backbone>\seed<42|43|44>\<key>\...`.
Manifest: `$CB\cross_backbones.resolved.json`. Báo cáo từng backbone tách riêng,
không lấy trung bình MetaFi++ và GraphPose-Fi để tạo một hàng “3 seeds”.

- T1: T-MPJPE, Target-PCK, clean MPJPE/PA-MPJPE và mean-positive DCE; mean ± sample SD ba training
  seeds. Clean reference không xếp hạng attack; no-trigger/clean-trained
  triggered controls giữ trong diagnostics.
- T2: sáu dose 0,.2,.4,.6,.8,1 × ba poison rates .1/.2/.4; mỗi rho một victim.
- T3: A1/A2/A3/A4 giữ factorial auxiliary carrier × dose coupling.
- DCE: vector residual `(pred_d-pred_0)-(target_d-true_0)` trên [2,3], mean năm
  dose dương trong seed rồi mean/sample SD giữa seeds. Dose0 không làm đẹp mean.
- Target-PCK: [2,3], threshold .2 × `norm(cleanGT[5]-cleanGT[12])`, không
  alignment; không phải clean PCK hay pose-level ASR.
- Đơn vị error mm; native attack operators/cover counts không có nghĩa cùng
  realized L2/Linf. Không claim equal-noise budget.

Defense hiện có vẫn chỉ là HPE-Li adaptation. Không gắn bảng defense HPE-Li
thành kết quả đã đánh giá trên hai backbone mới. Chưa chạy held-out subjects,
environments hay dataset thứ hai chỉ vì batch này có thêm kiến trúc.

## Kiểm thử local và giới hạn

155 test mới đạt (151 trong lượt tổng +4 worker end-to-end ở lượt riêng),
138 regression T1--T3 cũ đạt; các lượt đều exit0. Bao gồm model thật CPU,
source-operator parity, optimizer/scheduler/RNG resume, accumulation và nhóm
cuối thiếu mẫu, checkpoint atomic, paired NPZ replay và helper Windows parse/mock.
Bốn worker test dùng vòng lặp50epoch trên **fixture300TRAIN/3TEST và mạng nhỏ**,
không phải full MM-Fi. Không đăng ký task thật hoặc launch job trên Vân khi test.

Input audit giữ ordered file/frame identities và action hash như pipeline cũ,
không hash toàn bộ nội dung từng raw CSI/pose file. Chưa xác nhận clean utility,
VRAM CUDA thực, tốc độ hoặc số tấn công mới. Bảng chỉ complete khi đủ39victim
được audit cho từng backbone; test pass không thay cho kết quả thực nghiệm.
