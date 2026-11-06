# BackTime P3S3 resume + A1 dose 1% rồi 5%, máy Vân

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent (implementation requested by user).
- Date: 2026-11-07, Asia/Bangkok.
- Status: execution adapter; local CPU/toy tests only. Full MM-Fi, CUDA capacity,
  runtime, convergence and attack efficacy are NOT verified locally.
- User approved concurrent execution rather than waiting, and the temporary
  pause/resume of P3S3 after checkpoint checks; never restart BackTime from scratch.
- Local release gate: **363 passed** (46 shared-adapter tests,51 original
  two-split adapter tests,266 native/scientific/eval/loader regressions).
  Includes exact Windows toy-process hold/resume/stop, failure cleanup, two
  concurrent toy children, sequential queue, native50epoch CPU/toy train and
  six-dose/PCK replay/cache reuse for both new rates.13 warnings are CPU-only
  pin-memory notices. PowerShell syntax parse passed. No remote training or
  pause was performed; actual real-input CUDA smokes remain mandatory.

## Phạm vi cố định

- P1S1 BackTime đã hoàn tất: xác minh native receipt/checkpoint/NPZ, không train lại.
- P3S3 BackTime: giữ nguyên old plan, source, input audit, generator/surrogate,
  optimizer/scheduler/RNG và checkpoint; dùng native resume. Chỉ đổi controller.
  Phần epoch chưa commit có thể phải replay; không hứa giữ các step chưa lưu.
- Một worker A1 **P1S1/HPE-Li/seed42**, lần lượt rho=.01 rồi .05. Đây là extension
  của T2 10%/20%/40%, không phải chạy dose trên P3S3. Không gộp hai TEST split.
- Full TRAIN/TEST=133056/33264, floor-poison=1330/6652. Mỗi rho một scratch victim,
  50 epoch/final49, SGD .001/momentum .9/wd0, B32, workers4, FP32 như T2 cũ.
  Trigger md_multicarrier/bend/eps.185, paired U(.2,1), pivot1->[2,3], Z40deg*d.
  Không train riêng theo từng dose, không ép các poison subsets phải nested.
- Native evaluation đủ d=0,.2,.4,.6,.8,1: T-MPJPE và DCE; summary DCE chỉ mean năm
  dose dương. Clean MPJPE/PA giữ nguyên. Replay checkpoint bổ sung Target-PCK@.2
  tại d=1 bằng evaluator cũ, không suy Target-PCK từ scalar T-MPJPE.
- `MMFI_A1_LOW_POISON_REFERENCE.json`: cấu hình chuẩn hóa và18 SHA scientific
  files đối chiếu với archive seed42 a1/rho0p2/rho0p4. Chỉ rho/profile và đường dẫn
  data/device được loại khỏi recipe comparison; không chỉnh scientific engine.
  Hai cell ID mới được đăng ký vào một **private evaluator module** để giữ nguyên
  công thức metric và canonical registry. Nguồn adapter mới được bind riêng.

## Máy và đường dẫn đã biết

Người dùng gửi process listing: controller21676, P3S3 worker37288, Python
`C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`, old checkout
`C:\Users\Van\code_backtime_4090_20261106`.
PID chỉ là quan sát lịch sử: script tự tìm lại và kiểm tra ownership tại thời
điểm chạy, **không hard-code PID để kill**.

Old output (default đã điền đúng từ process listing):

```text
U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1
```

Output riêng của controller/dose:

```text
U:\backdooranalog\runs\mmfi_shared_backtime_dose_s42_4090_v1
```

Không pull/sửa checkout BackTime đang chạy. Clone mã mới vào folder mới; dataset,
prior, environment vẫn ở DataHome cũ, checkpoints BackTime vẫn tại old output.
Kiểm tra actual output volume>=10GiB trước pause. Không xóa hoặc di chuyển dataset.

## Lệnh cho người dùng

Chỉ làm trên máy Vân, không mở thêm launcher/GPU job trong lúc handover:

```powershell
$R = "$env:USERPROFILE\code_shared_dose_4090_v1"
git clone https://github.com/toan31072006/backdoor-analog.git $R
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; do not stop the old job' }
$S = Join-Path $R 'remote_tools\Start-MMFiBackTimeDoseShared4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode SwitchAndLaunch
```

Nếu folder clone đã tồn tại, **không overwrite** và không dùng code cũ không rõ
commit. Kiểm tra commit hoặc chọn tên folder mới. Sau khi handover thành công,
`SwitchAndLaunch` không chạy lần hai vì nó không được quyền dừng worker mới.

Mode độc lập, dùng cùng `$S`:

```powershell
# Chỉ xem; không mở checkpoint đang ghi, không pause/CUDA/train:
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
# Nếu đã Pause thành công mà muốn chia bước:
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Prepare không tự full-train. SwitchAndLaunch là explicit user command kết hợp
Pause -> Prepare -> Launch, chỉ đi tiếp sau khi toàn bộ kiểm tra đạt. Launch có
thể resume chính các checkpoint đã bind sau crash; không auto retry.

## Cách handover và chốt an toàn

1. CPU-only validate old plan/source/Python environment, native gate receipts,
   P1S1 completion, old input receipts và ordered IDs/prior cho hai rho mới.
2. Tự tìm đúng một original P3S3 worker. Mở **exact Windows process handle**
   trước khi xác minh CIM command line, parent, executable/source hash, old
   root/home, ownership context checksum/PIDs/plan hashes. Refuse foreign jobs.
3. Chỉ explicit Pause/SwitchAndLaunch mới được tạm giữ process đó. Giữ process
   trong lúc đọc committed checkpoint để không cản native Windows atomic writer.
   Kiểm tra native state/checksum/bindings, copy backup qua cùng một file handle,
   verify SHA, rồi mới terminate chính worker đã bind. Nếu bước kiểm tra/backup
   lỗi, finally cho original worker chạy tiếp; không terminate nó.
4. Đợi old controller tự drain/thoát và OS locks được giải phóng. Không kill
   controller/peer. Không xóa run.lock. Validate checkpoint cuối sau pause.
   Cửa sổ launcher cũ có thể báo failure/exit130 do handover có chủ đích;
   đối chiếu `PAUSED_AND_BACKED_UP` và tiến trình của **launcher shared mới**.
   File `.saving` chưa commit chỉ được **rename cùng folder** thành
   `.saving.paused_<uuid>`, ghi lại từng target. Không restore checkpoint cũ,
   overwrite staging hoặc silently dùng artifact dở.
5. Concurrent capacity test bằng real TRAIN: original BackTime B64 search/backward
   + B32 SGD smoke và A1 B32/2 native SGD steps (4poison/28clean). Các model được
   giữ resident qua barrier hai worker; tất cả smoke weights đều discard.
   Nếu OOM/guard/input mismatch: fail-stop, giữ log và checkpoint. Không shrink
   batch, AMP, CPU fallback hay auto-retry. BackTime sau permanent stop vẫn ở
   checkpoint đã backup; cần xem log rồi quyết định tiếp, không auto restart.
6. Launch hai worker sở hữu rõ `backtime`/`dose`, allowlist chỉ đúng hai PID này.
   Controller không tạo CUDA context thứ ba. Shared receipt không sửa old gate.
   Nếu một worker lỗi, không kill worker kia, không launch replacement; root
   locks được giữ đến khi peer thoát. Dose không bắt đầu rho5 nếu đã thấy peer
   failure. Interrupt trước checkpoint đầu tiên của dose sẽ refuse epoch0 fallback.

Giữ terminal foreground và Windows thức. Process suspension chỉ trong đoạn
backup; nếu terminal/controller bị cưỡng chế kill đúng đoạn đó thì có thể cần
manual recovery, không mở launcher khác hoặc xóa lock để bypass. Gửi log để
xác định trạng thái; chưa có quyền tự restore/terminate một PID mới.

## Artifacts / giới hạn

- New root: `shared.plan.json`, `pause_identity.json`, `pause_intent.json`,
  `pause.json`, `pause_backups/<uuid>/checkpoint.pt`, `shared.gate.json`,
  `executions/<action_uuid>/{workers.json,backtime/dose.stdout/stderr.log}`,
  `launch_resume_points/`, `dose/rho0p01`, `dose/rho0p05`, `evaluations/`,
  `results.json`. Không overwrite original receipts/checkpoints.
- Native BackTime tiếp tục xuất ở old `p3s3/`; master `results.json` chỉ được
  hoàn tất bằng cả hai split đã native-audit complete. Kết quả dose ở new root.
- Không sửa PDF/LaTeX hoặc tự thêm số mới vào bảng. Chưa có kết quả nghiên cứu mới
  chỉ vì launcher/tests đã sẵn sàng; 41% GPU utilization không chứng minh còn VRAM.
- Chỉ một seed42; concurrency có thể giảm tốc. Không khẳng định bitwise parity
  giữa execution schedules/GPU, hay gọi đây là tái lập forecasting BackTime gốc.
