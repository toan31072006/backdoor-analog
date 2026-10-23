# WiPose HPE-Li: dừng hai job cũ, giữ checkpoint và tiếp tục bản fast

Phiên bản này tiếp tục **Clean và A1 riêng** từ checkpoint epoch hoàn tất gần
nhất của output parallel cũ. Quyền dừng hai job WiPose và đổi sang bản fast là
yêu cầu mới của người dùng; quy tắc lịch sử không migrate trong hướng dẫn
parallel cũ không áp dụng cho lần nâng cấp có lineage này. Không sửa binding,
manifest, source hoặc checkpoint cũ để làm chúng mang profile mới.

Đây vẫn là pilot HPE-Li 18 khớp/2D từ nhánh DT-Pose, chưa phải reproduction
HPELi++ mới nhất hay bảng paper hoàn chỉnh. Seed42, 50epoch/final49,
physicalbatch32, SGDlr.01/momentum.9/wd0, FP32/workers0/drop_lastTrue,
ordinaryMPJPE, ordered full TRAIN/TEST, normalization, poison, target và sáu
dose/paired controls giữ nguyên. Sai số báo bằng đơn vị ảnh gốc, không gọi mm;
Target-PCK không phải ASR. Sau khi cả hai train xong, evaluate Clean rồi A1.

Bản fast cache **đúng tensor FP32 TRAIN đã normalize** trong RAM, tối đa
2GiB payload mỗi process, và gom kiểm tra gradient hữu hạn trên device. Không
đổi physicalbatch, AMP, TF32, loader workers, poison draw hay SGD. RAM tổng
của hai worker còn bao gồm model, optimizer và overhead ngoài cache. Gate hai
SGDstep B2 đối chiếu bitwise trên device thật trước update production, khôi phục
model/optimizer/RNG sau gate. Local CPU parity không chứng minh tốc độ CUDA,
VRAM/RAM fit hoặc hiệu quả attack. Chưa có hệ số tăng tốc hoặc ETA đo trên4090.

Các đường dẫn mặc định trên máy Vân:

| Vai trò | Đường dẫn/tên |
| --- | --- |
| Output cũ, chỉ đọc | `%USERPROFILE%\backdooranalog\runs\wipose_hpeli_a1_s42_parallel_v1` |
| Task cũ cần dừng | `WiPose-HPELi-Parallel-Seed42` |
| Backup mới mỗi lần `-Stop` | `%USERPROFILE%\backdooranalog\backups\wipose_parallel_before_fast\<timestamp_guid>` |
| Output fast mới | `%USERPROFILE%\backdooranalog\runs\wipose_hpeli_a1_s42_fast_v1` |
| Task fast mới | `WiPose-HPELi-Fast-Seed42` |
| Manifest fast | `wipose_hpeli_fast.resolved.json` |
| Provenance chuyển tiếp | `resume.lineage.json` |

## 1. Pull bản bổ sung trước khi dừng

Dùng Anaconda PowerShell trên máy Vân, không CMD. Các file runner/data/model/
helper cũ giữ nguyên bytes; pull thêm file mới không đổi source binding của job
cũ. Nếu pull lỗi, giữ mọi file và gửi lỗi, không reset hoặc ghi đè manifest.

```powershell
cd "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; chưa dừng hoặc launch.' }
```

Không tự sửa code cũ đang được checkpoint ràng buộc. Nếu checkout hiện tại có
thay đổi ở file cũ khiến pull không fast-forward, cần xử lý thay đổi đó trước.

## 2. Kiểm tra rồi dừng đúng hai job WiPose cũ

Lệnh đầu chỉ audit, không tạo backup hay dừng bất kỳ task/process nào:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Stop-WiPoseHPELiParallelTask.ps1
if ($LASTEXITCODE -ne 0) { throw 'Audit dừng chưa đạt; giữ job và gửi lỗi.' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Stop-WiPoseHPELiParallelTask.ps1 -Stop
if ($LASTEXITCODE -ne 0) { throw 'Chưa xác nhận dừng an toàn; chưa launch fast, giữ backup và gửi lỗi.' }
```

Helper xác thực current SID, task action encoded/config đúng output cũ và
wrapper của helper cũ, source SHA cùng manifest. Cả Clean và A1 phải có
`checkpoint.pt` không rỗng. Nếu có file `.saving`, helper từ chối và yêu cầu
chạy lại sau khi writer hoàn tất, không tự loop hay xóa file đó.

Khi có `-Stop`, helper tạo thư mục backup mới: `before_stop` chứa bản sao
manifest/source/poison/checkpoints và `hashes.json`, xác minh copy trước khi
stop task. Sau đó nó tìm process mới qua CIM, chỉ dừng Python đúng executable
trong env, đúng runner/output và current SID; controller trước, worker sau.
PID, CreationDate và command được đọc lại ngay trước mỗi stop. PID trong log
cũ không được dùng. WiFlow/PiW/MM-Fi và các Python khác không thuộc phạm vi.

`after_stop` giữ checkpoint atomic cuối cùng cùng hashes/bytes đã kiểm. Nếu
epoch vừa hoàn tất giữa audit và stop, bản sau stop có thể mới hơn bản trước.
Output cũ và `run.lock` giữ nguyên, không xóa lock. Phần giữa epoch chưa save
sẽ mất và được chạy lại từ checkpoint cuối; helper **không force-save** phần
đang chạy. Chỉ tiếp tục khi có `STOPPED_OLD_WIPOSE_PARALLEL` và exit0.

## 3. CPU preflight và chạy nền bản fast

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiFastTask.ps1 -JobsPerGpu 2 -PreflightOnly
if ($LASTEXITCODE -ne 0) { throw 'Fast CPU preflight lỗi; chưa launch.' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiFastTask.ps1 -JobsPerGpu 2
if ($LASTEXITCODE -ne 0) { throw 'Fast launcher lỗi; xem log trước khi gọi lại.' }
```

`-DryRun` chỉ in kế hoạch. `-PreflightOnly` audit full data, model/operator và
checkpoint/source/config/runtime cũ trên CPU; cả hai mode không migrate hoặc
ghi output mới. Helper launch tự chạy dry-run và CPU preflight lại trước đăng
ký task. Có thể mất thời gian hash dữ liệu mà chưa có Task/cell log; không gọi
helper lần hai trong lúc preflight còn foreground.

Launch mặc định truyền rõ `--resume-from` output parallel cũ; không fallback
sang scratch khi thiếu checkpoint. Controller thật khóa đọc output/cell cũ,
xác thực cả hai nguồn rồi mới tạo checkpoint mang profile mới và
`resume.lineage.json` trong output fast. Model, optimizer, RNG và số epoch cũ
được giữ; bước tiếp theo là epoch sau checkpoint cuối. Manifest/checkpoint cũ
giữ bytes/hash, lineage lưu provenance thay cho sửa hash trong manifest cũ.

Task S4U/Limited/Hidden không giới hạn thời gian, wrapper và conda ẩn cửa sổ,
độc lập SSH khi task đã bắt đầu. Máy Vân phải bật và không sleep/shutdown.
Helper chặn task/process WiPose serial/parallel/fast còn chạy cùng dataset,
kể cả output khác, và probe root/cell locks. Không đồng thời gọi nhiều helper
hoặc bypass guard. Tối đa hai worker độc lập; `-JobsPerGpu 1` là lựa chọn
explicit khi đã idle, không phải tự fallback sau OOM. Không automatic retry.

Nếu đổi đường dẫn khi chạy trước đây, truyền cùng `-DataHome`, `-DatasetRoot`,
`-Outdir`, `-EnvironmentPrefix`, `-ActionNpy` cho Stop; truyền `-ResumeFrom`
là **output cũ** và `-Outdir` là **output fast khác** cho Start. Giữ cấu hình
này qua các lần resume. Không đặt output mới bên trong output cũ hoặc ngược lại.

## 4. Xem tiến độ và audit kết quả

```powershell
$WIP_FAST = "$env:USERPROFILE\backdooranalog\runs\wipose_hpeli_a1_s42_fast_v1"
Get-ScheduledTask -TaskName WiPose-HPELi-Fast-Seed42 `
  -ErrorAction SilentlyContinue | Select-Object TaskName,State

foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$WIP_FAST.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Encoding UTF8 -Tail 15 }
}
foreach ($stage in @('clean','a1')) {
  $WIP_CELL_LOG = "$WIP_FAST\hpeli\seed42\$stage\console.log"
  Write-Host "`n$stage"
  if (Test-Path $WIP_CELL_LOG) {
    Get-Item $WIP_CELL_LOG | Select-Object FullName,LastWriteTime
    Get-Content $WIP_CELL_LOG -Encoding UTF8 -Tail 12
  }
}
```

`Running` không đủ chứng minh tiến triển. Xem epoch/timing trong hai cell log;
gate hoặc cache có thể chưa tới epoch đầu tiên. Hai `TRAIN_COMPLETE` chưa phải
full kết quả: còn Clean evaluation, A1 evaluation, six-dose/paired controls,
rồi `PILOT_COMPLETE`. TEST không chọn checkpoint hoặc epoch.

```powershell
$WIP_PY = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
& $WIP_PY -u .\ATKBackd\run_wipose_hpeli_fast.py --outdir "$WIP_FAST" --status
if ($LASTEXITCODE -ne 0) { throw 'Status audit lỗi; gửi output.' }

# Sau PILOT_COMPLETE: replay read-only artifact, không retrain/inference.
& $WIP_PY -u .\ATKBackd\run_wipose_hpeli_fast.py --outdir "$WIP_FAST" --check-only
if ($LASTEXITCODE -ne 0) { throw 'Artifact replay chưa đạt; không gọi là complete.' }
Get-Content "$WIP_FAST\wipose_pilot.md" -Encoding UTF8
Import-Csv "$WIP_FAST\a1_six_doses.csv" | Format-Table -AutoSize
```

Khi task fast thật sự đã dừng, dùng lại đúng helper/output/profile để resume.
Giữ cả nguồn cũ, backup, dữ liệu, env và lineage. Không xóa checkpoint/log hoặc
sửa manifest để vượt mismatch. Local test CPU/mock chỉ xác nhận các gate và
parity trong phạm vi được test; chưa là full-data CUDA run hoặc kết quả TEST mới.
