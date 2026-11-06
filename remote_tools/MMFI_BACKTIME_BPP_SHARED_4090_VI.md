# BppAttack-CSI P3S3 / HPE-Li / seed42, cùng BackTime trên máy Vân

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent; setup requested by user.
- Date: 2026-11-07, Asia/Bangkok.
- Status: local implementation/tests only; no remote pause/training performed.
- Requested row: fresh BppAttack-CSI/MM-Fi P3S3/HPE-Li/seed42. User reports
  poison-rate jobs complete while BackTime P3S3 still runs with spare GPU.
- This is one added exploratory T6/comparison row, not a full T1 table or the
  complete CVPR2022 BppAttack paper training. No manuscript/PDF changes.
- Local gate: **205 passed** (37 new Bpp adapter/owner/gate/process tests,
  46 previous shared-adapter tests,122 source/native/eval regressions).
  Includes native50epoch CPU/toy Bpp training, six-dose/PCK replay/cache and
  interruption/resume with identical model/optimizer/RNG/cover state; two actual
  toy child processes overlap/drain.13 warnings are CPU pin-memory notices.
  PowerShell parsed; ATKBackd and all6556c43 active adapters remain unchanged.
  This is not a full-data or CUDA-capacity result; remote Prepare is mandatory.

## Recipe cố định

- Giữ đúng `_method_config(...,'bpp_source')` của bảng cũ, chỉ chuyển official
  split sang protocol3/s3 và thêm provenance. Không đổi operator/loss/budget.
- TRAIN240570 / TEST80190,810/270 sequences; train E01--E03, test E04,
  subject/sequence/frame disjoint; split seed0, experiment seed42.
- Một scratch HPE-Li victim50epoch, fixedfinal49, B32, workers0,
  SGDlr.001/momentum.9/wd0, ordinary MPJPE, FP32; không AMP/scheduler,
  pretrained/clean warm-start, epoch selection bằng TEST hoặc giảm batch.
- Rho=.1 =>24057 poisoned recipients; thêm24057 clean-label covers disjoint.
  Tổng20% modified-input recipients, KHÔNG phải tất cả đều dirty-label poison.
  Không gọi native comparison là equal-realized-noise/equal-modified-budget.
- BppAttack-CSI là biến thể BackdoorBench đã dùng trong bài:8 mức,
  no dithering, quantization interpolation theo dose,10% residual negatives.
  Fixed cover IDs qua các epoch, lazy donor có hoàn lại từ toàn bộ TRAIN.
  TRAIN-only donor pool/key và stochastic cover RNG phải khớp khi resume.
- Đây không phải đầy đủ PGD/SupCon training của paper BppAttack. Giữ attribution
  và các adaptation/deviation từ `attack/static_source_baselines.py`;
  BackdoorBench pin `f02e3534645f0ee63d6848653062cd6c0d6c400d`.
- Paired train doseU(.2,1), targetpivot1->[2,3], Z40deg*d; đánh giá full TEST
  d=0,.2,.4,.6,.8,1. T-MPJPE và DCE từng dose; mean DCE chỉ năm dose dương;
  CleanMPJPE/PA và Target-PCK@.2,d1, scale cleanGT(5,12).
  Canonical evaluator chỉ thay counts và identity callback trong private module;
  metric formulas/replay tolerance<.05mm không đổi. Có same-victim no-trigger
  control, per-sample NPZ. Không suy Target-PCK từ scalar error.
- Không train thêm Clean/A1/baseline khác. Clean-trained reference/trigger
  control chưa được bind vào row này; report ghi rõ, không gọi full T1 complete.

## Vì sao cần handover dù GPU còn dư

BackTime đang dùng allowlist PID cố định. Worker dose đã kết thúc không làm PID
mới của BppAttack tự được cho phép. Process Bpp mở ngoài controller có thể làm
BackTime fail ở phase boundary dù còn VRAM. Không mở job Bpp bằng tay/bypassguard.

Adapter mới private-compose coordinator đã test ở6556c43, không sửa scientific
`ATKBackd` hoặc các launcher/receipt đang chạy. Roles mới là `backtime`/`bpp`,
không giả Bpp là P1S1 hoặc là dose worker.

Chỉ explicit `Pause`/`SwitchAndLaunch` mới bàn giao. Script tự tìm original
two-split worker hoặc worker BackTime của dose-shared controller. Xác minh
exact Windows handle, Python/source, parent command, context hash, master/gate/
input binding. Với dose-shared case, cả1%/5% phải có final49 checkpoint,
RNG/evaluation/NPZ bindings hợp lệ và worker dose phải đã thoát.

Sau xác minh, tạm giữ đúng P3S3, validate/copy committed checkpoint vào output
mới và verifySHA, rồi mới terminate worker đã bind. Nếu validate/backup lỗi thì
resume process cũ. Parent tự drain/thoát; không kill parent/peer hay xóa lock.
Native checkpoint giữ optimizer/scheduler/RNG; epoch chưa commit có thể replay,
không restart từ đầu. Orphan `.saving` chỉ rename cùng folder và ghi receipt,
không xóa/restore checkpoint cũ tự động. Không dùng lại Switch sau handover.

Nếu BackTime đã tự hoàn tất khi chạy lệnh: phải có native-complete final49
receipt/checkpoint/NPZ và GPU idle, rồi ghi `ALREADY_COMPLETE_NO_PROCESS_STOPPED`.
Không stop process nào hoặc retrain BackTime. Nếu không có live worker mà P3S3
chưa native-complete, refuse; không âm thầm recovery/scratch restart sau crash.
Capacity smokes vẫn discard, production worker BackTime chỉ xác minh/export
checkpoint đã xong; chỉ Bpp có production training mới.

Prepare chạy hai real-TRAIN CUDA capacity smokes cùng lúc: BackTime B64/B32,
Bpp B32/2SGDsteps với4poison/4cover/24clean. Giữ model resident qua barrier,
discard toàn bộ smoke weights. Fail/OOM thì giữ artifacts, không retry/shrink/
AMP/fallback. GPU utilization còn dư không bảo đảm đủ VRAM hoặc nhanh hơn.

## Cách chạy trên máy Vân

Không pull/overwrite checkout đang chạy. Clone sang folder mới, dùng environment
cũ; output trên U: được check actual free>=10GiB. Defaults đã điền đúng old root:

```text
U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1
```

Lệnh dưới gồm handover, Prepare và Launch, chỉ tiếp tục khi checks đạt:

```powershell
& {
    $R = "$env:USERPROFILE\code_bpp_p3s3_4090_v1"
    git clone https://github.com/toan31072006/backdoor-analog.git $R
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed; old job unchanged' }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$R\remote_tools\Start-MMFiBackTimeBppShared4090.ps1" -Mode SwitchAndLaunch
}
```

Không tự Ctrl+C old job trước. Nếu muốn chỉ xem, đổi Mode thành `Inspect`:
không mở checkpoint đang ghi, không pause/CUDA/train. `Pause`, `Prepare`,
`Launch` cũng tách riêng được. Sau Pause thành công, chỉ dùng Prepare/Launch
cho các bước tiếp; không chạy SwitchAndLaunch lần nữa để stop một PID mới.

Giữ terminal mới mở và Windows thức. Terminal cũ có thể báo failure do handover
có chủ đích: kiểm tra `PAUSED_AND_BACKED_UP` và log controller mới. Nếu crash
đúng đoạn tạm giữ process thì có thể cần manual recovery; không tự xóa lock,
copy checkpoint hoặc mở launcher khác. Gửi cuối log để xác định trạng thái.

## Outputs và giới hạn

```text
U:\backdooranalog\runs\mmfi_shared_backtime_bpp_p3s3_s42_4090_v1\
  shared.plan.json / pause*.json / pause_backups / shared.gate.json
  executions\<prepare|launch_uuid>\workers.json
  executions\<...>\backtime.stdout.log / bpp.stdout.log / *.stderr.log
  bpp_source\checkpoint.pt / eval_cache.json / scientific_audit.json
  evaluations\bpp_source\evaluation.json / per_sample.npz
  results.json
```

BackTime tiếp tục ở old P3S3 output. P1S1 và hai dose cũ không train lại, không
overwrite. Cell/evaluation artifacts1%/5% vẫn ở root dose-shared cũ; receipt
handover mới lưu hash của cả hai. Old combined shared-results có thể chưa được
finalize vì parent cũ đã bàn giao; không coi điều đó là mất hai cell hoàn tất.

Launch chỉ allowlist đúng hai child PIDs; controller không tạo CUDA context
thứ ba. Failure dừng launch mới, drain healthy peer; không kill/retry peer.
Một seed42, không across-seed SD/CI. Local CPU/toy tests không phải số đo mới
trên P3S3/RTX4090, hay bằng chứng attack efficacy/tái lập paper gốc.
