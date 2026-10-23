# WiPose / HPE-Li: hai job train Clean + A1 seed42

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: user-requested implementation + local validation.
- Origin Date: 2026-10-23, Asia/Bangkok.
- Verification Status: LOCAL_VALIDATED_NOT_FULL_WIPOSE_OR_GPU_RESULT.
- Version Label: wipose_hpeli_a1_s42_parallel_v1.
- Người dùng yêu cầu dùng hai job trên RTX4090 sau khi được giải thích bản
  serial hiện chỉ có một worker. Quyền push sau test tiếp nối bộ WiPose đã duyệt;
  không phải quyền kill job khác, thuê Vast, xóa dữ liệu hoặc đổi recipe.

## Phạm vi

Đây là thay đổi **lịch thực thi**, không thay phương pháp/tham số. HPE-Li WiPose
18 khớp/2D, seed42, scratch Clean và scratch A1, 50epoch/final49, physicalbatch32,
SGDlr.01/momentum.9/wd0, ordinaryMPJPE, FP32/workers0/drop_lastTrue giữ nguyên.
Input normalization, ordered lists/next-frame filter, poison/target/epsilon và
sáu dose/paired controls giữ nguyên nhánh serial. Không full DT-Pose, không
dataset3D, không báo sai số ảnh thành mm và không có kết quả TEST mới từ local.

Hai process train độc lập: mỗi process có model/optimizer/BN/RNG seed42 riêng,
không chia một victim thành hai job và không load Clean weights vào A1. Sau khi
**cả hai train xong**, controller chấm Clean, rồi chấm A1 với Clean checkpoint
và paired no-trigger/clean-model+trigger controls. TEST không chọn checkpoint.

- Sidecar: `ATKBackd/run_wipose_hpeli_parallel.py`; không sửa serial runner,
  preparer, dataset reader, model hoặc nguồn PiW/MM-Fi/WiFlow lịch sử.
- Output riêng: `runs/wipose_hpeli_a1_s42_parallel_v1`.
- Manifest: `wipose_hpeli_parallel.resolved.json`, profile parallel riêng.
- Task: `WiPose-HPELi-Parallel-Seed42`.
- Không import/migrate checkpoint serial hoặc sửa binding cũ để resume vào bản
  mới. Nếu serial đang train, không launch thêm bản parallel; gửi log trước.
- 4090/24GB có thể là ứng viên cho hai job, nhưng chưa đo peak VRAM/throughput
  trên dữ liệu thật. Hai job không bảo đảm nhanh gấp đôi. Không tự giảm batch,
  bật AMP hoặc retry khi OOM; log lỗi và checkpoint còn được giữ.

## 1. Hoàn tất dữ liệu trước

Theo `WIPOSE_HPELI_A1_VI.md`: download -> extract -> prepare. Raw count đúng
132847TRAIN/33753TEST. Nếu prepare đang chạy, giữ nguyên process/cửa sổ của nó.
Bản bổ sung không đổi `prepare_wipose.py`/`wipose_data.py`, nên không làm mất
source binding của lượt prepare. MATv7.3 cần mat73/h5py trong env trước manifest.

```powershell
cd "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; giữ file, chưa launch.' }

$WIP_HOME = "$env:USERPROFILE\backdooranalog"
$WIP_ROOT = "$WIP_HOME\datasets\WiPose_processed_v1"
if (!(Test-Path "$WIP_ROOT\preparation.json")) {
    throw 'Prepare chưa COMPLETE. Không launch, không xóa output.'
}
Test-Path "$WIP_HOME\actions\data_bend.npy"
nvidia-smi
```

Nếu còn process prepare/preflight, không chạy lại. `preparation.json` là mốc
hoàn tất preparation, không phải mốc train/evaluate. Helper còn audit nội dung.

## 2. Chạy nền tối đa hai job train

Dùng Anaconda PowerShell trên **máy Vân**, không CMD. Không dán dấu nhắc PS.

```powershell
cd "$env:USERPROFILE\code"
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiParallelTask.ps1 -JobsPerGpu 2
```

Helper dry-run rồi preflight CPU/full hashing trước đăng ký Task. Có thể chưa
thấy Task/log trong thời gian này: không launch lặp. Khi helper trả về, kiểm
controller log có hai dòng `TRAIN clean`/`TRAIN a1` và cell log bắt đầu tiến
triển. `Running` một mình không chứng minh tiến triển.

Helper không dừng process khác. Nó từ chối task/process serial hoặc parallel
WiPose đang dùng cùng dataset mà nó nhận diện được, ngay cả output khác, và
probe root/cell locks của output parallel. Đây không phải khóa nguyên tử toàn
máy cho mọi launcher tùy ý: không đồng thời mở serial helper và parallel helper,
không bypass guard hoặc xóa lock. Một controller quản lý tối đa hai train worker.

Task được đăng ký S4U/Limited/Hidden theo user hiện tại, độc lập SSH. Sau khi task
thật sự bắt đầu, có thể ngắt SSH/tắt máy local. **Máy Vân phải luôn bật, không
sleep/shutdown.** Trong lúc prepare hoặc helper preflight còn foreground, đóng
SSH có thể ngắt chúng; chờ launcher trả về trước.

Nếu OOM/worker lỗi: không retry, không launch cell còn pending; controller chờ
worker còn sống do nó sở hữu hoàn tất rồi báo lỗi. Nó không kill peer đang train.
Một worker `TRAIN_COMPLETE` chưa phải kết quả A1/pilot hoàn tất.

## 3. Kiểm log

Mở PowerShell khác trên Vân:

```powershell
$WIP = "$env:USERPROFILE\backdooranalog\runs\wipose_hpeli_a1_s42_parallel_v1"
Get-ScheduledTask -TaskName WiPose-HPELi-Parallel-Seed42 `
  -ErrorAction SilentlyContinue | Select-Object TaskName,State

foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$WIP.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Encoding UTF8 -Tail 15 }
}
foreach ($stage in @('clean','a1')) {
  $WIP_LOG = "$WIP\hpeli\seed42\$stage\console.log"
  Write-Host "`n===== $stage ====="
  if (Test-Path $WIP_LOG) {
    Get-Item $WIP_LOG | Select-Object FullName,LastWriteTime
    Get-Content $WIP_LOG -Encoding UTF8 -Tail 10
  } else { Write-Host 'Chưa có cell log; xem controller/preflight.' }
}
```

Mốc theo thứ tự: hai `TRAIN_COMPLETE` -> `EVAL_COMPLETE clean` ->
`EVAL_COMPLETE a1` -> `PILOT_COMPLETE`. Có thể hai train kết thúc khác thời điểm;
controller không dùng TEST để cắt ngắn victim chậm hơn.

## 4. Audit, kết quả và resume

```powershell
cd "$env:USERPROFILE\code"
$WIP = "$env:USERPROFILE\backdooranalog\runs\wipose_hpeli_a1_s42_parallel_v1"
$WIP_PY = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
& $WIP_PY -u .\ATKBackd\run_wipose_hpeli_parallel.py --outdir "$WIP" --status

# Chỉ sau PILOT_COMPLETE; read-only, không train/inference.
& $WIP_PY -u .\ATKBackd\run_wipose_hpeli_parallel.py --outdir "$WIP" --check-only
if ($LASTEXITCODE -ne 0) { throw 'Audit chưa đạt; gửi output, không train lại mù.' }
Get-Content "$WIP\wipose_pilot.md" -Encoding UTF8
Import-Csv "$WIP\a1_six_doses.csv" | Format-Table -AutoSize
```

Checkpoint atomic mỗi epoch, kèm optimizer/RNG/runtime/execution/config/input
bindings. A1 training checkpoint không chứa Clean SHA (độc lập scratch); A1
evaluation vẫn ràng buộc Clean result/checkpoint/NPZ và tái tính controls.
`training.complete.json` khóa SHA final49checkpoint và poison plan, trạng thái
`TRAINED_NOT_EVALUATED`; thiếu evaluation không được báo pilot complete.

Nếu task thật sự đã dừng, chạy lại **đúng helper/outdir** để explicit resume.
Cell train đã hoàn tất được xác thực và bỏ qua; final49 thiếu marker được phục
hồi marker sau input audit, không optimizer step thêm. Evaluation bị ngắt được
chạy lại từ checkpoint final, không retrain. Giữ nguyên data/source/runtime;
không tự sửa manifest, xóa lock/cache hoặc copy checkpoint serial sang.

`-JobsPerGpu 1` cũng được hỗ trợ trong cùng profile để người dùng chọn sau khi
job lỗi/dừng; đây là execution setting, không đổi scientific recipe hoặc tự
fallback. Chỉ thay khi không còn controller/worker của output đó. `-CleanOnly`
là tùy chọn riêng; hai job không tạo thêm seed hay biến partial thành full table.

## Kiểm thử và giới hạn

Gate tổng ngày23/10: **293 passed**, exit0 (228.58s), gồm94test mới
(46parallelrunner +48helper) và199test hồi quy WiPose serial/data/model/metrics.
CPU/synthetic kiểm exact optimizer/RNG resume, parity số học với serial, A1 train
trước Clean, evaluation dependency, artifact tamper, scheduler capacity/failure
drainage và PowerShell task/guard parsing. Hai subprocess toy OS thật đã xác nhận
overlap/lifetime. Hai process HPE-Li WiPose thật cũng qua SGDstep CPU/synthetic
batch32, output32x1x18x2, gradients finite, parameters update. Review độc lập
không thấy lỗi chặn sau sửa guard đổi1/2slot khi task đã idle.

Không gọi CPU/synthetic/mocks là đo VRAM, tốc độ, khả năng chạy full WiPose trên
4090 hoặc attack efficacy. Nguồn serial/preparer/data/model và các run lịch sử
không sửa; chưa launch task/GPU trên Vân từ workspace này.
