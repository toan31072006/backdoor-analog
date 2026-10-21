# WiFlow MM-Fi: train thật Clean + A1 seed42

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Ngày: 2026-10-22. Người dùng yêu cầu train thật, sau test commit/push.
- Origin Mode: implementation + CPU synthetic reproducibility validation.
- Verification Status: CPU_SYNTHETIC_VALIDATED; FULL_MMFI_CUDA_UNVERIFIED.
- Version Label: mmfi_wiflow_clean_a1_seed42_pilot_v1.
- Scope: MM-Fi protocol1-s1, split seed0, TRAIN133056 / TEST33264, seed42.
- Đây là pilot Clean + A1, không phải full T1/T2/T3, defense hoặc dataset2.
- Không dừng/kill task Graph, MetaFi++ hoặc job khác; không sửa dữ liệu/kết quả
  lịch sử. Assistant chưa truy cập được GPU trên Vân từ workspace này.

## Recipe khóa trước khi xem kết quả

Model/loss author source:
[WiFlow MM-Fi 3D](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling/blob/8ddb9a01772c368fc591f69e8414044b49d817e1/cross_dataset_test/WiFlow/wiflow.py),
commit `8ddb9a01772c368fc591f69e8414044b49d817e1`, Apache-2.0.
Kiến trúc/forward/source model giữ nguyên; không load pretrained checkpoint.
Trích riêng definitions, không execute toàn bộ training script bên thứ ba.

- AdamW lr1e-4, wd1e-4, betas(.9,.999), eps1e-8; clip gradient norm1.
  Đây là scalar optimizer/default LR của code tác giả được đối chiếu, không
  lấy biến `weight_decay=1e-5` ở main vốn không được optimizer của họ dùng.
- Effective batch64; physical32 x accumulation2, phần cuối sample-weighted và
  flush theo epoch. BN dùng physical32, không tuyên bố giống physical64.
- **Common ordinary MPJPE**, không native SmoothL1 + bone loss. Native bone
  edges joint9 khác payload joint8 đã được ghi nhận, nhưng loss đó không chạy
  trong pilot này. Không âm thầm sửa topology/model hoặc dùng poison mask/dose
  để cho victim loss quyền attacker mạnh hơn.
- **Fixed LR**, không ReduceLROnPlateau/early stopping/selection theo PCK
  validation của tác giả. Giữ full official TEST để evaluation sau training;
  không dùng TEST chọn LR, strength, checkpoint hoặc threshold.
- 50 epochs cả Clean/A1, fixed final49; FP32, workers0, một WiFlow slot;
  checkpoint atomic mỗi epoch, RNG/optimizer được resume cùng recipe/output.
- A1 rho=.1; poison uniform, paired U(.2,1), pivot1/joints[2,3], Z40d.
  Clean rho0 và cùng-seed A1 trigger diagnostic. Giữ toàn bộ CSI preprocessing,
  raw-domain injection, pose units, action prior và split của A1 đã khóa.
- Sáu dose0,.2,.4,.6,.8,1; Target-PCK@.2 GT scale(5,12), T-MPJPE@1,
  Clean MPJPE/PA và mean-positive DCE. Giữ ordered identities, poison/input/
  source/checkpoint/NPZ hashes, no-trigger target controls và raw JSON cache.

Đây là **WiFlow architecture adapted to the common MM-Fi backdoor benchmark**,
không reproduction full recipe hay accuracy của paper tác giả. Clean utility
phải được kiểm tra trên dữ liệu thật trước kết luận cross-backbone; không
claim attack mạnh từ một pipeline test tổng hợp.

## Chạy trên Vân

Mở **Anaconda PowerShell**, không CMD. Graph phải đã pause đúng worker nếu
vẫn đang chạy; helper này không tự làm việc đó. Suspend Graph giữ VRAM/lock.
MetaFi++ + một WiFlow worker là hai active training jobs; không mở thêm hai
WiFlow jobs. Không mặc định GPU rảnh chỉ vì utilization có lúc bằng0.

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
cd "$env:USERPROFILE\code"
git status --short
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull chưa thành công; dừng, không reset/xóa file.' }
nvidia-smi

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1WiFlowTask.ps1
```

Nếu đã chép ZIP probe cũ vào repo, `git pull` có thể báo untracked files would
be overwritten. **Dừng và gửi output**; không `git clean`, reset, xóa file hoặc
giải nén `-Force`. Không cần chuyển checkpoint Meta/Graph sang WiFlow.

Đây không phải lệnh `probe_wiflow_mmfi.py`. Task mới
`MMFI-A1-WiFlow-Seed42`; output mới
`backdooranalog/runs/mmfi_wiflow_a1_s42_v1`. Clean chạy xong rồi A1, không song
song hai cell WiFlow. MetaFi++ vẫn là process/task riêng. Đóng SSH/VSCode không
phải tắt task; không sleep/shutdown. Không auto-retry hoặc tự shrink batch OOM.

Chỉ kiểm plan, không ghi output/task/data/GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1WiFlowTask.ps1 -DryRun
```

Muốn chỉ kiểm clean trước, thêm `-CleanOnly`; sau khi clean task đã kết thúc,
gọi lại helper không `-CleanOnly` để authenticate/reuse clean và chạy A1.
Resume sau gián đoạn dùng lại **đúng helper/output**; task/worker còn sống thì
helper từ chối mở trùng. Không xóa lock. Mất điện chỉ mất phần epoch chưa được
checkpoint; Resume Process chỉ dành cho process đang suspend, không process chết.

Linux/Vast về sau (một worker, phải có đúng dữ liệu/action/env):

```bash
python -u ATKBackd/run_a1_wiflow.py --data-home /path/to/backdooranalog \
  --outdir /path/to/new/mmfi_wiflow_a1_s42_v1 --device cuda:0
```

## Log và kiểm hoàn tất

Mở PowerShell khác trên Vân:

```powershell
$WF = "$env:USERPROFILE\backdooranalog\runs\mmfi_wiflow_a1_s42_v1"
Get-ScheduledTask -TaskName MMFI-A1-WiFlow-Seed42 | Select-Object TaskName,State
foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$WF.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Encoding UTF8 -Tail 15 }
}
Get-ChildItem "$WF\wiflow\seed42\*\console.log" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
  ForEach-Object { Get-Content $_.FullName -Encoding UTF8 -Tail 15 -Wait }
```

`-Wait` ở log clean không tự chuyển sang log A1; dùng loop dưới để theo dõi
MetaFi++ và WiFlow đồng thời. Ctrl+C chỉ dừng cửa sổ xem log.

```powershell
$CB = "$env:USERPROFILE\backdooranalog\runs\mmfi_cross_backbones_s42_43_44_v1"
$WF = "$env:USERPROFILE\backdooranalog\runs\mmfi_wiflow_a1_s42_v1"
while ($true) {
  Clear-Host
  Get-Date
  foreach ($item in @(
    @{name='MetaFi++'; path="$CB\metafiplusplus\seed42\*\console.log"},
    @{name='WiFlow'; path="$WF\wiflow\seed42\*\console.log"}
  )) {
    Write-Host "`n===== $($item.name) =====" -ForegroundColor Cyan
    $log = Get-ChildItem $item.path -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log) {
      Write-Host "$($log.FullName) | $($log.LastWriteTime)"
      Get-Content $log.FullName -Encoding UTF8 -Tail 8
    } else { Write-Host 'Chua co log.' }
  }
  Start-Sleep -Seconds 10
}
```

Kiểm hoàn tất read-only sau khi hai cell xong:

```powershell
$WF = "$env:USERPROFILE\backdooranalog\runs\mmfi_wiflow_a1_s42_v1"
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_a1_wiflow.py --outdir "$WF" --check-only
Get-Content "$WF\wiflow_pilot.md" -Encoding UTF8
```

Ready không tự chứng minh thành công; cần hai dòng VERIFIED, exit0, completion
JSON và kết quả pilot. Không thêm mean/SD vì chỉ một training seed.

## Kiểm thử và giới hạn

Source SHA/AST/gradient parity được test ở suite probe. Runner mới kiểm
manifest/source/runtime tamper, one-slot worker entrypoint, no-write dry-run,
weighted accumulation, RNG/optimizer resume và interrupted atomic checkpoint.
Worker tests chạy loop50epoch với model source thật lẫn tiny regressor trên
fixture **12 TRAIN/3 TEST**, native A1 poison thật; đây không phải full MM-Fi.
Task helper được parse/mock; không đăng ký task thật khi test.

Kiểm thử cuối: **166 test riêng biệt đạt theo các nhóm**, exit0:
40 mới (34 runner +6 PowerShell helper) và126 source/probe/cross-backbone/
accumulation regression. Test model thật physical32 được chạy thêm đạt, không
đếm thành seed hoặc test riêng thứ167. Online pinned source SHA/AST audit đạt;
SHA của model/license/wrapper trong Git index cũng khớp source passport.
Không thay scientific sources HPE-Li/MetaFi++/GraphPose-Fi.

Các nhóm lệnh kiểm thử (đặt MKL_THREADING_LAYER=SEQUENTIAL, OMP/MKL threads2
trên Windows; chọn basetemp mới cho mỗi lượt):

```text
python -m pytest ATKBackd/tests/test_a1_wiflow_runner.py ATKBackd/tests/test_a1_wiflow_task.py -q
python -m pytest ATKBackd/tests/test_wiflow_mmfi_probe.py ATKBackd/tests/test_a1_cross_backbones.py ATKBackd/tests/test_a1_cross_backbone_adapter.py ATKBackd/tests/test_a1_cross_backbone_eval.py ATKBackd/tests/test_a1_cross_backbone_worker.py ATKBackd/tests/test_a1_backbone_accumulation.py -q
python ATKBackd/probe_wiflow_mmfi.py --audit-upstream
```

Chưa launch task/GPU trên Vân hoặc train full133056/33264 trong workspace
local. Chỉ sau khi người dùng pull/chạy và audit artifact thực mới được gọi
là có kết quả MM-Fi. Test pass không chứng minh convergence/efficacy.
