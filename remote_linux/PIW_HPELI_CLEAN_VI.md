# Person-in-WiFi-3D: HPE-Li Clean seed 42 trước, chưa A1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: run / preflight; implementation + local synthetic validation.
- Origin Date: 2026-10-23, Asia/Bangkok.
- Verification Status: LOCAL_TESTS_ONLY_NOT_FULL_PIW_RUN.
- Version Label: piw_hpeli_clean_seed42_v1.
- Người dùng đồng ý kiểm thử, sau đó commit/push launcher để tự pull trên Vân.
  Clean trước; A1 chỉ chạy sau khi adapter 14 khớp được kiểm thử riêng.
- Đây là **một clean scientific victim**, không phải bảng attack/defense, không
  phải bằng chứng efficacy hay mean ± SD nhiều seed. Chưa có clean accuracy
  hoặc GPU ETA của full PiW run từ các test local.
- Không sửa hoặc dừng job WiFlow, MetaFi++, GraphPose-Fi, MM-Fi hay dữ liệu/
  checkpoint lịch sử. Một PiW worker bên cạnh một WiFlow worker là hai active
  training jobs; controller/conda wrapper không phải thêm victim.

## Phạm vi và recipe khóa trước run

Dataset processed được người dùng tải từ Google Drive file ID
`1tr98LAenpHUxhV3ibq91Qkjk6cxsftik`. Nguồn tải không tự bảo đảm schema/split:
preflight vẫn phải kiểm tra các file trước khi train. Không tải lại hoặc ghi
đè dữ liệu vì một check thất bại.

- Dataset root mặc định:
  `backdooranalog/datasets/Person-in-WiFi-3D_processed_v1`.
- Dùng **one-person** của official TRAIN/TEST list; không biến dữ liệu nhiều
  người thành một pose trung bình. TRAIN giữ điều kiện next-frame hiện có,
  TEST không áp filter next-frame. Profile khóa ordered identities; dự kiến
  TRAIN 27,876 và TEST 2,586 mẫu sau chọn lọc, không phải toàn bộ file trong ZIP.
  Đây là split file được cung cấp, có capture prefixes chung giữa TRAIN/TEST;
  không gọi là held-out session, subject hoặc environment generalization.
- Input CSI `[3,180,20]`: amplitude 90 hàng đầu và phase 90 hàng sau được
  min-max riêng theo pipeline PiW hiện có. Không dùng normalization MM-Fi.
  Pose có 14 khớp × 3 tọa độ; đơn vị metres đã khai báo trong profile, không
  tự đoán hoặc đổi đơn vị dựa trên giá trị TEST.
- HPE-Li 14-joint, random initialization; ordinary MPJPE, không attack loss.
  AdamW lr `0.001`, weight decay `0.01`, batch `32`, 200 epochs, FP32,
  workers `0`; fixed LR, không scheduler/AMP/clipping/early stopping.
  LR `0.001` là adaptation theo benchmark WBackdoor đã chọn; không phải
  reproduction nguyên bản nhánh DT-Pose/PiW dùng LR `0.01`.
- Chỉ checkpoint **final epoch 199** được đánh giá trên TEST. Không chọn best
  checkpoint theo TEST hoặc tune recipe sau khi xem TEST. Partial training
  không được ghi thành kết quả full clean pilot.
- Clean MPJPE và Clean PA-MPJPE được báo theo mm trên toàn bộ 14 khớp.
  Không có T-MPJPE/Target-PCK/DCE hay attack result trong nhánh Clean này.
- Source/config/runtime/input fingerprints, checkpoint, ordered prediction
  artifact và result được giữ riêng. Checkpoint atomic mỗi epoch và resume
  cùng output/manifest; không skip batch, tự giảm batch hoặc retry khi OOM.

**Khác dataset/khác skeleton** nên không so trực tiếp mm giữa PiW14 và MM-Fi17
như cùng test set. Nhánh A1 MM-Fi hiện tại không tự trở thành A1 PiW chỉ bằng
đổi tên dataset; target joints, skeleton, normalization, trigger domain,
scale và metric controls cần port/audit trước.

## Preflight: kiểm trước khi tạo Task

Chạy trên **máy Vân, Anaconda PowerShell**, không CMD. Chỉ pull sau khi người
chuẩn bị code xác nhận commit/push đã xong. Nếu pull báo conflict/untracked,
dừng và giữ nguyên file; không `git clean`, reset hoặc xóa checkpoint.

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
cd "$env:USERPROFILE\code"
git status --short
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; chưa chạy thêm job.' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiCleanTask.ps1 -DryRun
```

`-DryRun` chỉ đọc metadata/code/runtime để resolve plan; không đọc dataset,
không ghi output, không dùng CUDA hoặc inspect/create Task.

Muốn kiểm data/model trước mà **chưa train hoặc tạo Task**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiCleanTask.ps1 -PreflightOnly
```

Preflight đọc list, kiểm tên/cặp/shape/finite values và lưu kích thước byte +
SHA256 của **tất cả listed one-person CSI/keypoint**: 28,121 TRAIN và 2,586
TEST identities, mỗi identity hai file. Bao gồm cả 245 TRAIN identities bị
loại khỏi training vì thiếu next-frame. Các hàng nhiều người vẫn có trong
hash của official list nhưng không được train hoặc chấm metric. Sau đó chạy
CPU forward/backward của model; không optimizer update, không CUDA, không
tạo experiment output. Có thể tốn vài phút do nhiều file nhỏ. Dừng khi báo
thiếu/mismatch, không tự tạo list hoặc bỏ mẫu để qua gate.

Normal launch chạy dry-run + preflight trước tạo Task. Controller audit lại
input trước manifest, private worker audit trước training: **ba lượt hash
startup** (helper, controller, worker), không phải ba GPU jobs hoặc nhân số
epoch. Final audit và `--check-only` cũng đọc/hash lại để xác thực dữ liệu.
Trong giai đoạn startup có thể chưa có cell console log; xem terminal/helper
và controller log, không launch trùng.

## Chạy nền trên Vân

Kiểm trạng thái hiện tại trước khi thêm đúng **một PiW job**:

```powershell
nvidia-smi
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'run_a1_wiflow|run_a1_cross_backbones|run_piw_hpeli_clean' } |
  Format-List ProcessId,ParentProcessId,CommandLine
```

GPU-Util 50% là một snapshot, không có nghĩa bảo đảm còn 50% công suất hoặc
đủ memory cho job mới. Một Graph đang suspend vẫn giữ RAM/VRAM. Helper không
tự pause/kill/resume bất kỳ job nào để nhường tài nguyên; nếu OOM, giữ log và
không lặp launch hay sửa batch trong manifest đã khóa.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-PiWHPELiCleanTask.ps1
```

- Task: `PIW-HPELi-Clean-Seed42`, current Windows account, S4U/Limited,
  không execution time limit, không automatic retry.
- Output: `backdooranalog/runs/piw_hpeli_clean_s42_v1`.
- Manifest: `piw_hpeli_clean.resolved.json`.
- Đóng SSH/VSCode không yêu cầu dừng Task; không sleep/shutdown máy. Đây không
  phải bảo đảm sống qua mất điện hoặc user khác kill process.
- Helper từ chối nếu cùng output có task/process/khóa đang active. Không xóa
  `run.lock`. Task name của account/action/output khác không bị ghi đè.
- `-DatasetRoot`, `-DataHome`, `-Outdir`, `-EnvironmentPrefix` hỗ trợ đường dẫn
  local riêng. Khi đổi DataHome mà không truyền các path còn lại, helper lấy
  dataset/output/env từ DataHome mới. Không đổi recipe/runtime để resume kết
  quả cũ; mismatch phải audit hoặc tạo output mới có chủ đích.

## Log và kiểm tra hoàn tất

Mở PowerShell khác trên Vân:

```powershell
$PIW_OUT = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_clean_s42_v1"
Get-ScheduledTask -TaskName PIW-HPELi-Clean-Seed42 | Select-Object TaskName,State

foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$PIW_OUT.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 20 }
}

$PIW_LOG = "$PIW_OUT\hpeli\seed42\clean\console.log"
if (Test-Path -LiteralPath $PIW_LOG) {
  Get-Item -LiteralPath $PIW_LOG | Select-Object FullName,LastWriteTime
  Get-Content -LiteralPath $PIW_LOG -Encoding UTF8 -Tail 20 -Wait
} else {
  Write-Host 'Chưa có cell log; kiểm tra controller stdout/stderr và preflight trước.'
}
```

Ctrl+C ở cửa sổ xem log chỉ dừng theo dõi, không dừng train. Log có thể chỉ
cập nhật khi hết epoch; im lặng ngắn không chứng minh sập. `Running` không
chứng minh tiến triển; `Ready` cũng không tự chứng minh kết quả thành công.

Status đọc artifact hiện có; full audit dùng `--check-only` (hash data và
prediction, không repair/train/evaluate lại trên GPU):

```powershell
$PIW_PY = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
& $PIW_PY -u .\ATKBackd\run_piw_hpeli_clean.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_OUT" --status

& $PIW_PY -u .\ATKBackd\run_piw_hpeli_clean.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_OUT" --check-only

if (Test-Path -LiteralPath "$PIW_OUT\clean_pilot.md") {
  Get-Content -LiteralPath "$PIW_OUT\clean_pilot.md" -Encoding UTF8
}
```

Kết quả full clean pilot nằm ở `clean_result.json` và `clean_pilot.md`.
Cell giữ `checkpoint.pt`, `evaluation.json`, `clean_predictions.npz` và
`console.log`. Chỉ dùng kết quả khi completion/audit thành công, không chỉ vì
hai file Markdown/JSON tồn tại. Một seed không có sample SD giữa các seed.

Trường hợp hiếm: process bị kill **sau khi cả `evaluation.json` và
`clean_result.json` đã hoàn tất nhưng trước khi xuất `clean_pilot.md`**. Sau
khi xác nhận cùng task/worker đã dừng, dùng check read-only trước, rồi chỉ
xuất lại report dưới root lock:

```powershell
& $PIW_PY -u .\ATKBackd\run_piw_hpeli_clean.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_OUT" --check-only
if ($LASTEXITCODE -ne 0) { throw 'Audit chưa đạt; không export hoặc sửa artifact.' }

& $PIW_PY -u .\ATKBackd\run_piw_hpeli_clean.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$PIW_OUT" --export-only
if ($LASTEXITCODE -ne 0) { throw 'Export chưa thành công; giữ lại lỗi để kiểm tra.' }
Get-Content -LiteralPath "$PIW_OUT\clean_pilot.md" -Encoding UTF8
```

`--export-only` xác thực lại input/checkpoint/NPZ/metrics rồi ghi report; không
model inference, GPU hay training. Nếu completion JSON còn thiếu thì trả
incomplete, không tự dựng kết quả hoặc coi checkpoint là đã chấm TEST.

Nếu user khác kill process hoặc mất điện, sau khi xác nhận không còn cùng
task/worker active, chạy lại **cùng helper/output**. Resume từ checkpoint epoch
đã lưu, phần epoch chưa save phải chạy lại. Không xóa lock/checkpoint, không
đổi optimizer/batch/dataset nhằm vượt lỗi. Giữ thông báo lỗi để audit.

Atomic writes có retry hẹp khi Windows tạm khóa file: chỉ lặp cùng thao tác
rename của temporary file đã serialize, tối đa 5 lần tổng cộng, chờ
0.05/0.10/0.15/0.20 giây giữa các lần. Chỉ áp dụng `PermissionError` Windows
5/32/33; lỗi khác hoặc hết số lần sẽ báo lỗi, checkpoint trước vẫn được giữ.
Đây **không phải retry job, batch, optimizer update hoặc epoch**; không thay
đổi policy không automatic retry của launcher.

## A1 sau Clean

Launcher này **không tự queue A1**, không training poison, không xuất metric
attack. Sau clean sanity, implementation A1 PiW là bước tiếp theo với profile
14 khớp riêng và đủ controls/dose/metric. Clean checkpoint có thể được dùng
làm clean reference sau khi audit tương thích; không hứa tự động reuse bất
kỳ checkpoint hoặc biến clean weights thành backdoored model.

## Kiểm thử local trước handoff (2026-10-23)

- Runner: 49 test đạt; helper: 14 test đạt. Helper tests dùng Task/process
  mocks, không đăng ký Task thật. Runner tests dùng fixture synthetic; có
  vòng 200 epoch bằng model nhỏ và so khớp chính xác checkpoint/RNG/optimizer
  giữa chạy liên tục và resume sau epoch 97.
- HPE-Li thật đã qua CPU forward/backward và một bước AdamW ở batch32:
  input `[32,3,180,20]`, output `[32,1,14,3]`, 833770 parameters, gradient
  finite và weights thay đổi. Không dùng dữ liệu TEST thật cho test này.
- Đã kiểm tra từ chối dữ liệu/manifest/artifact bị đổi, incomplete status,
  replay metric từ NPZ, recovery khi thiếu completion marker, cùng bảo toàn
  checkpoint trước khi atomic write thất bại.
- Main-agent integration gate: 128 passed /1 failed; cả 63 test PiW mới đạt.
  Test lỗi thuộc runner WiFlow cũ: Windows `WinError 5` ở atomic rename khi
  ghi checkpoint liên tục trên ổ E. Cùng test chạy riêng với temp trên ổ C
  đạt (1 passed); không sửa source/checkpoint/config WiFlow để qua gate.
  Vì vậy không ghi lượt integration ban đầu thành một lượt all-green.
- Môi trường local: Python3.13.5, torch2.9.1+cpu, numpy2.1.3. Vân dùng môi
  trường riêng; dry-run/preflight trên Vân vẫn là gate bắt buộc. **Chưa chạy
  full processed PiW hoặc CUDA trên Vân, chưa có clean accuracy hay A1 result.**
