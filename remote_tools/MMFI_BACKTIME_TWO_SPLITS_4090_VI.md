# BackTime-only / HPE-Li / P1S1 + P3S3 / Windows RTX4090

## Material Passport

- User decision2026-11-06: train lại từ đầu trên máy Windows4090 của Vân,
  hai pipeline độc lập P1S1 và P3S3, seed42. Không resume/import checkpoint3090.
- academic-research-suite / experiment-agent; execution setup and local tests.
- Source base: GitHub `toan31072006/backdoor-analog`, commit
  `798723db02f93edb4312b6f3bde9f2f450d99262`.
- BackTime upstream commit `9e0a1f9deecbacfdfea9159e6e8892148b44da9f`.
  Giữ nguyên `attack/backtime_mmfi.py` của bản nguồn đã published;
  đây là CSI/pose/dose adaptation, **không tái lập forecasting nguyên bản**.
- Chỉ chọn BackTime; không chạy lại SIBA/LADDER/A1/defense hoặc backbone khác.
  Source/input/plan/GPU/runtime bind riêng. Không sửa hoặc xóa run lịch sử.
- Verification: chưa chạy full MM-Fi/CUDA4090 từ workspaceAdmin. Local CPU/toy
  tests không chứng minh VRAM, tốc độ, hội tụ hay hiệu quả tấn công.
- Local gate2026-11-06: **389 passed**, gồm37tests của adapter mới và352tests
  hồi quy. HPE-Li thật chạy BackTime fit2epoch/B64 +SGD B32 trên CPU/synthetic;
  hai OS process toy overlap/drain, PowerShell parse +Inspect thật, kiểm tra
  split/manifest/recipe/runtime guards đạt. Không phải kết quả training MM-Fi.
- Storage follow-up gate cùng ngày: **403 passed** (51adapter +352regression),
  gồm14tests mới cho output khác DataHome, low-space và Windows volume-type
  guards. Source BackTime/scientific runner không đổi; chưa chạy Prepare trên U:.

## Lịch và metric được giữ

| Split | TRAIN | TEST | Poison recipients |
|---|---:|---:|---:|
| P1S1 |133056|33264|13305|
| P3S3 |240570|80190|24057|

Mỗi split chạy tuần tự **Clean reference50 → BackTime attacker100/B64 →
victim mới50/B32 → evaluation**, trong khi hai split chạy song song trên
cuda:0. Tổng4scientific reference/victim trainings và2attacker schedules,
không phải chỉ hai model50epoch. Clean reference không được dùng warm-start
BackTime surrogate hoặc final victim.

- HPE-Li victim/Clean: SGDlr.001/momentum.9/wd0,50epoch/final49, ordinary
  MPJPE, physicalB32, workers0, FP32. Không AMP/fallback/early stopping/bestTEST.
- BackTime: surrogate scratchAdam.0001,100fullTRAIN passes/B64, epoch0..10
  clean (11passes),89alternating passes; generatorAdam.005,λ.05,
  history6/trigger4, graph+GCN+constant branch, anchored replacement.
  Error-driven recipients giữ onlineSmoothL1 warmup, không đổi thành uniform.
  Giữ mỗi recipient một generator update: P1S1 **1184145**, P3S3 **2141073**.
  Không bank nhỏ, cắt epoch hoặc gộp recipient để làm nhanh.
- Rho.1; pose targetpivot1→joints[2,3],Z40°×dose; TRAINpairedU(.2,1).
  Six-dose TEST0,.2,.4,.6,.8,1; Target-PCK@.2 scale cleanGT(5,12).
  T-MPJPE/Target-PCK@1,mean-positiveDCE,CleanMPJPE/PA, same-victim no-trigger
  and clean-model+trigger controls +per-sample artifacts được giữ.
- P1S1 dùng official per-action subject splitseed0; không gọi globally
  held-out subject. P3S3 giữ E01–E03/S01–S30 train và E04/S31–S40 test.
  Không gộp hai TEST set/backbone thành hai seed hay suy ra SD từ frame.
- Native ε.2 là bound quanh historical anchor ở đơn vị TRAINstandardized,
  không phải bound thay đổi thực tế so với CSI gốc. Không equal-noise claim.

## Chạy trên máy Vân

Dùng checkout riêng chứa launcher mới, không pull/extract đè checkout đang
train. Dataset/env/prior reuse ở `C:\Users\Van\backdooranalog`; source mới
không kèm rawMMFi/env/checkpoints. Không phải chỉ copy một file launcher.

Người dùng đã cho phép test rồi commit/push riêng launcher. Clone vào thư mục
mới, đối chiếu `rev-parse HEAD` với commit bàn giao; không pull đè checkout đang
chạy. Không cập nhật code sau Prepare. Giữ thay đổi coauthor/công việc khác.

```powershell
$Repo = 'C:\Users\Van\code_backtime_4090_20261106'
if (Test-Path -LiteralPath $Repo) { throw 'Folder exists: preserve it; choose another NEW checkout path.' }
git clone --depth 1 --branch main https://github.com/toan31072006/backdoor-analog.git $Repo
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; do not continue.' }
git -C $Repo rev-parse HEAD
$S = Join-Path $Repo 'remote_tools\Start-MMFiBackTimeTwoSplits4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
```

Inspect chỉ đọc plan, không tạo output/load dataset/CUDA. Mặc định chọn env
`backdooranalog\envs\dose-backdoor\python.exe`, rồi
`envs\mmfi-p3\Scripts\python.exe`; muốn đổi thì truyền cùng `-PythonExe`
cho mọi mode. Không tự cài/nâng dependency. Dataset phải ở
`datasets\Compress`, prior `actions\data_bend.npy` đúng authenticSHA đã pin.

### C: ít trống: giữ data/env, lưu output trên ổ khác

User báo C: còn2GiB, U: còn60.36GiB. Không suy ra2GiB đủ cho hai full runs.
`-Outdir` có thể ở ổ local khác; không cần copy dataset/env/actions sang U:.
Ngưỡng10GiB được kiểm tra trên **ổ chứa output**, không còn chặn theo DataHome.
Đây là entry guard, không phải bảo đảm full-run tuyệt đối không hết dung lượng.

Vì Prepare cũ đã dừng trước audit/CUDA, có thể cập nhật checkout chưa chạy rồi
chọn root mới trên U:. Không làm bước này khi đang có worker sử dụng checkout.

```powershell
git -C $Repo pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; preserve local changes.' }
$O = 'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -Outdir $O
```

Chỉ sau Prepare thành công, dùng cùng `-Outdir $O` cho CheckOnly/Launch/Status:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly -Outdir $O
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -Outdir $O
```

Output phải là thư mục mới prefix`mmfi_backtime_` dưới folder`runs`. Windows
phải nhận ổ output là fixed/removable disk, còn trống ít nhất10GiB; ổ network
mapped/UNC hoặc ổ không thấy từ Python bị từ chối trước khi tạo output. U: thực
tế chưa được assistant kiểm tra trên máy Vân; nếu nó là network/cloud drive,
không bypass guard. Không xóa hoặc di chuyển run cũ trên C:, không đổi DataHome.
Ổ C: vẫn cần khoảng trống cho hoạt động của Windows; lưu output sang U: không
phải bảo đảm hệ điều hành ổn định nếu C: tiếp tục đầy.

Các lệnh không có `-Outdir` bên dưới chỉ dùng output mặc định trên DataHome.
Nếu chọn U: thì dùng các lệnh `$O` ở trên cho mọi mode, không Prepare mặc định
lần nữa. Khi máy không còn compute job khác, chạy riêng **Prepare**, rồi dừng
xem log:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

Prepare audit/hash đủ hai split rồi mở hai process đồng thời. Capacity smoke
dùng64realTRAIN frames/split với sourceBackTime fit2epoch (warmup0) để thực
hiện B64surrogate backward +6generator updates; hai fitter gặp barrier khi
model đang resident. Sau đó B32SGD2updates. Tất cả trọng số smoke bị bỏ,
không ghi checkpointtraining; **không đổi production100epoch/warmup10**.
Smoke không bảo đảm full-run tuyệt đối, nhưng phải đạt thực tế trên4090.
OOM/mismatch thì dừng, giữ log; không tự fallbackbatch, AMP, serial hoặc retry.
Gate bind SHA của cả hai input/poison manifests, source/resolved plan và GPU/
runtime; đổi dữ liệu hoặc recipe sau smoke sẽ bị từ chối, không tự ghi đè gate.

Chỉ sau `PREPARED_ONLY`, xem gate rồi Launch riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

**Giữ SSH/PowerShell mở, máy bật và không sleep.** Không tạo TaskScheduler,
service hay boot/autoretry. Controller foreground giữ root lock và hai PID
worker riêng. Chỉ hai PID nó vừa spawn được phép chia sẻ GPU; không chấp nhận
mọi `python.exe` hay kill job người khác. Worker lỗi: không tạo replacement,
chờ peer đang chạy kết thúc (drain), giữ log và báo lỗi; không kill peer.
Controller Launch không khởi tạo CUDA context thứ ba; từng worker tự xác minh
GPU/runtime thật với gate trước khi tạo model training.

Output mới:

```text
backdooranalog/runs/mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_v1/
  two_splits.plan.json
  two_splits.gate.json
  executions/<prepare-or-launch-id>/<split>.stdout.log, <split>.stderr.log
  p1s1/{plan.json,inputs.json,poison.json,logs/,clean_surrogate/,backtime/,results.json,comparison.md}
  p3s3/{plan.json,inputs.json,poison.json,logs/,clean_surrogate/,backtime/,results.json,comparison.md}
  results.json
```

Xem tiến độ trong SSH thứ hai:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

`Status` không dùng CUDA hoặc tự launch. File/PID tồn tại không chứng minh
epoch tiến lên. `BACKTIME_P1S1_P3S3_COMPLETE` chỉ được in sau khi hai split
đã có final49/evaluation với binding hợp lệ. Chỉ một seed, không acrossseedSD.

Sau crash của **run mới4090** có thể Launch thủ công cùng root/source/env
để resume state hợp lệ; không đổi root thành run3090 hay import checkpoint.
Nếu muốn scratch lần nữa thì chọn root mới có prefix`mmfi_backtime_` bằng
`-Outdir`, chạy lại Prepare. Không xóa lock/gate/.saving hay sửa manifest.

## Phạm vi thay đổi source

Runner source-adapted cũ chỉ được tổng quát hóa phần chọn data-config,
count/sequence checks và nhãn báo cáo cho P1S1/BackTime-only. Mặc định P3S3
SIBA/BackTime cũ giữ nguyên. Optimizer, target, trigger fitting, FFT behavior,
model architecture, train/eval/metric và checkpoint serialization không sửa.
Adapter mới bind chính SHA của nó bên ngoài common source map. Không dùng
native serial exclusivity để launch hai queue riêng rồi tự bỏ guard.
