# Person-in-WiFi-3D: HPE-Li A1 seed 42, dùng Clean đã hoàn tất

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: run / preflight; implementation + local synthetic validation.
- Origin Date: 2026-10-23, Asia/Bangkok.
- Verification Status: LOCAL_TESTS_ONLY_NOT_FULL_PIW_A1_RUN.
- Version Label: piw_hpeli_a1_s42_v1.
- Người dùng yêu cầu chạy A1 sau Clean và đã cho phép push sau kiểm thử.
  Người dùng tự pull/chạy trên Vân; không có remote launch từ workspace Admin.
- Một seed là pilot, không có SD/CI giữa các seed và chưa đủ toàn bộ bảng bài báo.
  Không sửa/xóa/retrain Clean, WiFlow, MetaFi++, GraphPose-Fi hay run lịch sử.

## Phạm vi đã khóa

Đây là **A1 CSI/HPE adaptation cho PiW 14 khớp**, không đổi tên trigger PiW
`micro_dropper` cũ thành A1, không thay A1 MM-Fi đang có. Train một victim A1
mới từ scratch, dùng kết quả Clean đã xác thực làm đối chứng read-only.

- Dataset: `datasets/Person-in-WiFi-3D_processed_v1`, mirror Drive người dùng
  cung cấp (`1tr98LAenpHUxhV3ibq91Qkjk6cxsftik`), không tự tải/ghi đè dữ liệu.
- Official one-person lists: TRAIN 27,876 sau DT-Pose next-frame filter; TEST
  2,586 không lọc next-frame. Không lấy subset, tự tạo split hay chọn theo TEST.
  TRAIN/TEST có capture prefixes chung: không gọi là held-out session/subject.
- HPE-Li 14 khớp, seed 42, scratch; **200 epoch, final epoch 199**, physical
  batch 32, workers 0, FP32; AdamW lr `0.001`, wd `0.01`, ordinary MPJPE.
  Fixed LR, không AMP, scheduler, clipping, pretrained hay TEST checkpoint
  selection. Đây là recipe PiW benchmark đã chọn, không claim nguyên bản
  DT-Pose dùng lr `0.01` hay recipe MM-Fi 50 epoch/SGD.
- Poison rate `0.1`, uniform selection seed 42; paired trigger/payload dose
  `U(0.2,1)`, epsilon `0.185`, góc Z `40` độ, pivot `7`, target joints `[11,13]`.
  Không chọn target/góc/epsilon sau khi thấy TEST.
- Input PiW `[3,180,20]`: feeder min-max amplitude 90 hàng đầu và phase 90 hàng
  sau riêng như Clean. Áp dụng operator A1 không đổi trên amplitude đã normalize,
  với **3 nhóm × 30 subcarrier**, ghép lại đúng vị trí; giữ nguyên phase. Không
  giả định 90 subcarrier liên tiếp, không complex injection legacy và không
  normalize lần hai sau trigger. Đây là port domain/geometry, không method mới.
- Action prior mặc định `actions/data_bend.npy` phải có và được hash/audit.
  Không thay bằng random action, file khác hoặc tự tạo file khi thiếu.

## Clean reference và metric

Clean root mặc định: `runs/piw_hpeli_clean_s42_v1`. Preflight xác thực manifest,
checkpoint final 199, source/config/input binding và saved prediction/metrics;
không chỉ đọc scalar Markdown, không sửa hoặc train lại Clean. Không load Clean
weights làm initialization A1: hai victim độc lập từ scratch cùng recipe.

Người dùng đã gửi Clean MPJPE `92.8767` mm, PA-MPJPE `57.6072` mm và clean
PCK@0.2 `71.8263`%. Đây là số người dùng báo từ run Vân; không phải kết quả A1
mới và chưa được chạy lại trên dữ liệu thật từ workspace local. Authentication
trên máy Vân vẫn bắt buộc trước khi reuse.

Giữ nghĩa metric chính giống MM-Fi, đổi skeleton/scale đúng PiW:

- **T-MPJPE@1 (mm) ↓:** triggered prediction so với attacker target chỉ tại
  `[11,13]`, không Procrustes alignment.
- **Target-PCK@0.2@1 (%) ↑:** cùng hai target joints, khoảng cách tới attacker
  target ≤ `0.2 × ||GT[6]-GT[4]||` của clean GT từng mẫu. Không dùng clean PCK
  thay thế và không gọi joint Target-PCK là pose-level ASR.
- **Clean MPJPE / Clean PA-MPJPE (mm) ↓:** no-trigger prediction so với GT trên
  toàn bộ **14** khớp; alignment chỉ ở PA metric sạch.
- **DCE (mm) ↓:** norm của `(pred_d-pred_0)-(target_d-GT)` trên target joints,
  summary mean năm dose dương, không đưa d=0 vào để giảm trung bình.
- Đánh giá đủ `d=0,0.2,0.4,0.6,0.8,1`, same-victim no-trigger target control
  và clean-trained victim + trigger. Lưu per-sample predictions/identities để
  replay; low T-MPJPE hoặc high PCK riêng lẻ chưa chứng minh học backdoor.

PiW có 14 khớp và scale `(6,4)`, MM-Fi có 17 khớp và scale `(5,12)`. Có thể
giữ tên cột chính và trình bày thành hai dataset blocks, nhưng **không so tuyệt
đối các số mm/PCK như cùng một TEST set**. Tọa độ khai báo metres, báo mm ×1000;
không suy đoán units hoặc scale theo prediction để làm đẹp kết quả.

## Pull, kiểm tra và chạy một job trên Vân

Dùng **Anaconda PowerShell**, không CMD. Nếu git báo conflict hoặc untracked
collision thì dừng, giữ file để audit; không reset/clean/xóa run.

```powershell
cd "$env:USERPROFILE\code"
git status --short
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; chưa chạy thêm job.' }

nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiA1Task.ps1 -DryRun
```

`-DryRun`: chỉ resolve metadata/source/runtime, không data/CUDA/output/Task.
Muốn kiểm dữ liệu/Clean/action/model/metric nhưng chưa launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiA1Task.ps1 -PreflightOnly
```

Sau khi gate đạt, chạy nền:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiA1Task.ps1
```

Helper luôn dry-run rồi full read-only preflight trước đăng ký Task. Hash nhiều
file nhỏ/Clean artifacts có thể mất vài phút, lúc đó **chưa có Task/cell log**.
Đợi cửa sổ helper đầu tiên, không launch thêm vì chưa thấy task. Controller và
worker tiếp tục xác thực input trước khi train; nhiều lượt audit không phải
nhân đôi training. Preflight chỉ in poison count/hash và cấu hình; danh sách
private poison samples nằm trong manifest/`poison.json`, không in hàng nghìn
tên mẫu ra terminal. Kiểm tra CPU gồm cặp poison thật, dose0 identity, phase
không đổi và smoke test metric sáu dose; không dùng optimizer hay CUDA.

Clean reference yêu cầu cùng runtime/source/device và biến môi trường thread
đã ghi khi chạy Clean. Nếu báo mismatch, giữ nguyên manifest/checkpoint và gửi
lỗi; không sửa manifest, đổi package hoặc train lại Clean để vượt kiểm tra.

Các lượt audit không tạo thêm GPU job. Preflight failure dừng trước
registration, không bỏ qua gate.

- Task: `PIW-HPELi-A1-Seed42`, current SID, S4U/Limited/Hidden, không time limit.
- Output: `runs/piw_hpeli_a1_s42_v1`; manifest `piw_hpeli_a1.resolved.json`.
- Cell: `hpeli/seed42/a1/console.log`; checkpoint atomic mỗi epoch, optimizer/RNG.
- Chính xác **một PiW A1 worker**, có thể chạy cùng WiFlow độc lập nếu còn tài
  nguyên. Snapshot GPU 50% không đảm bảo còn đủ công suất/VRAM. Helper không
  dừng, suspend, retry OOM hoặc giảm batch của bất kỳ job nào.
- Task/config binding gồm code, data, dataset, env, output, Clean root và action.
  Task account/action/config khác không bị ghi đè. Root/cell locks được probe
  và recheck sau preflight; không xóa lock file vì file còn tồn tại.
- Thoát SSH sau khi helper tạo Task không yêu cầu dừng Task. Không sleep/tắt máy;
  không bảo đảm sống qua mất điện hay người khác kill process.

Có thể dùng các tham số `-DataHome`, `-DatasetRoot`, `-EnvironmentPrefix`,
`-Outdir`, `-CleanOutdir`, `-ActionNpy`, `-TaskName` cho đường dẫn local khác.
Không đổi recipe/source/data/environment để resume manifest đã khóa.

## Xem log, status và kết quả

Mở PowerShell khác trên Vân, không dán kèm dấu nhắc `PS ...>`:

```powershell
$PIW_A1 = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_a1_s42_v1"
Get-ScheduledTask -TaskName PIW-HPELi-A1-Seed42 -ErrorAction SilentlyContinue |
  Select-Object TaskName,State

foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$PIW_A1.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 20 }
}

$PIW_LOG = "$PIW_A1\hpeli\seed42\a1\console.log"
if (Test-Path -LiteralPath $PIW_LOG) {
  Get-Content -LiteralPath $PIW_LOG -Encoding UTF8 -Tail 20 -Wait
} else {
  Write-Host 'Chưa có cell log; xem terminal helper và controller logs trước.'
}
```

Ctrl+C ở cửa sổ theo dõi chỉ dừng tail. `Running` không tự chứng minh tiến
triển; `Ready` chỉ đúng là hoàn tất khi log/completion/audit xác nhận. Epoch
`199/199` là đã đủ training, vẫn phải chờ full six-dose TEST evaluation.

```powershell
cd "$env:USERPROFILE\code"
$PIW_PY = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
$PIW_A1 = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_a1_s42_v1"

& $PIW_PY -u .\ATKBackd\run_piw_hpeli_a1.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_A1" --status

& $PIW_PY -u .\ATKBackd\run_piw_hpeli_a1.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_A1" --check-only
if ($LASTEXITCODE -ne 0) { throw 'Audit chưa đạt; giữ log, không tự sửa artifact.' }

Get-ChildItem "$PIW_A1\*.md" -File |
  ForEach-Object { Get-Content -LiteralPath $_.FullName -Encoding UTF8 }
```

Nếu đã dùng nondefault paths khi launch, truyền lại đúng `--dataset-root`,
`--clean-outdir`, `--action-npy` lúc status/check/export. `--check-only` audit
dữ liệu và replay saved metrics, không train hay inference lại trên GPU. Sau
audit đạt, `--export-only` chỉ khôi phục report nếu JSON/NPZ completion đã có;
không dựng kết quả từ checkpoint chưa evaluation.

Nếu job bị kill, trước hết xác nhận task/worker cùng output không còn active,
rồi chạy lại **cùng helper/output/config**. Resume từ epoch đã save; phần epoch
chưa save phải chạy lại. Không xóa lock/checkpoint và không sửa manifest để
vượt mismatch. Không có automatic job retry hoặc tự retrain Clean.

Nếu evaluation đã có checkpoint/NPZ/metrics hợp lệ nhưng bị ngắt giữa hai
completion JSON hoặc trước khi xuất report, normal resume chỉ phục hồi
publication dưới khóa: không train/inference lại và không thay prediction NPZ.
Artifact sai hash vẫn bị từ chối; không tự coi checkpoint chưa chấm TEST là
kết quả hoàn tất. Atomic rename có bounded retry file-lock Windows như nhánh
Clean; đây không phải lặp job, epoch hay optimizer update.

## Kiểm thử và giới hạn

Các test local dùng CPU/synthetic và Task/process mocks; không đăng ký Task thật
hay chạy full processed PiW/CUDA trên Vân. Cần gate dữ liệu/provenance và clean
utility tại máy Vân; source test không đảm bảo attack efficacy. Không ghi số
A1, confidence interval hoặc tuyên bố đủ chuẩn bài báo khi chưa có run thật.

Gate cuối ngày 2026-10-23: **174 passed**, exit0, 339.76 giây. Gồm 57 test
adapter/source-parity/geometry/metric, 30 runner, 17 helper Task mock, 7 handoff
recovery và 63 regression của nhánh Clean. Hai assertion ở lượt tích hợp đầu
(102 passed/2 failed) đã sửa: raw input thay đổi bị Clean binding từ chối sớm
hơn test dự kiến, và thông báo dry-run không khớp chuỗi test. Lượt cuối mới là
gate all-green; không bỏ qua test lỗi hoặc sửa dữ liệu để vượt gate.

Fixture có 12 TRAIN retained, 3 TEST và **một poison pair thật**; train Clean
và A1 đủ200epoch bằng model nhỏ, so exact optimizer/weights/RNG khi ngắt sau
epoch97 rồi resume. Có kiểm tra Clean read-only, input/action/poison/NPZ/metric
tamper, cached no-GPU/no-inference và phục hồi publication bị ngắt nhiều lần.
Model HPE-Li thật kiểm thêm CPU batch32 với ba cặp poison synthetic: output
`[32,1,14,3]`, gradient finite, AdamW cập nhật weights, phase giữ nguyên.

Môi trường kiểm thử: Python3.13.5, torch2.9.1+cpu, numpy2.1.3;
MKL_THREADING_LAYER=SEQUENTIAL, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 chỉ áp dụng
cho test local. Không tự áp các biến thread này lên Vân nếu khác run Clean.
Không có full PiW A1 accuracy, CUDA benchmark hoặc remote launch trong gate.
