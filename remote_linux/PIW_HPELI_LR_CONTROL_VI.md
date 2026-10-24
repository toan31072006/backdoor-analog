# PiW: kiểm chứng riêng learning rate

Lượt kiểm chứng Clean + A1 seed42, cùng target hai hông của v2. Đây là
development experiment sau pilot, chưa phải bảng comparison đầy đủ hoặc blind
TEST. Giữ nguyên code, checkpoint và kết quả v1/v2.

| Thành phần | V2 đã chạy | LR control mới |
| --- | --- | --- |
| AdamW learning rate | **0.01** | **0.001** |
| Target | Hip7, joints11/13, 40°d quanh normalize(GT5−GT7) | Giữ nguyên |
| Training | 200 epoch/final199, B32, drop_last=True, workers0, FP32 | Giữ nguyên |
| Optimizer/loss | AdamW wd.01, betas(.9,.999), eps1e-8; ordinary MPJPE | Giữ nguyên |
| Data/trigger/poison/metric | Official split, normalized AP, eps.185, rho.1, paired U(.2,1), PCK@.2 | Giữ nguyên |

**Chỉ LR là thay đổi khoa học.** Không quay lại `drop_last=False`, đổi trục,
góc, epsilon, poison rate hoặc ngưỡng PCK trong lượt này. Profile/output/source
binding và tên task mới phục vụ tách provenance, không thay training algorithm.

V2 do người dùng cung cấp: tại dose1 T-MPJPE407.39(no-trigger)→205.99mm;
Target-PCK0.12→29.68%; DCE173.61 so với stationary384.05mm. Các đối chứng này
cho thấy steering so với cùng victim không trigger. Clean v2 đạt154.38mm,
kém run v1 92.88mm, nhưng chưa chứng minh LR là nguyên nhân. V1/v2 đã thay cả
recipe và target; LR control này kiểm chứng một yếu tố từ v2.

Train **hai victim Clean/A1 mới từ scratch**, cùng recipe LR.001, không warm-start
từ v1/v2 hoặc dùng Clean cũ làm control. Seed42, shuffle/drop_last và cố định
final199 giữ nguyên. Hai role có thể train song song; full TEST chỉ chạy sau khi
cả hai hoàn tất. Không chọn epoch/seed/target/LR tiếp theo theo điểm TEST; báo
đây là development-selected control, không phải tuning hoàn toàn độc lập TEST.

## PowerShell trên máy Vân

Code trong `C:\Users\Van\code`; dataset/action/env giữ nguyên. Task mới:
`PiW-HPELi-LRControl-Seed42`. Output mới:
`runs\piw_hpeli_body_axis_s42_lr001_v3`. Không gọi launcher v2 vào output này.

```powershell
cd "$env:USERPROFILE\code"
$S = "$env:USERPROFILE\code\remote_tools\Start-PiWHPELiLRControlTask.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -File "$S" -Mode DryRun -JobsPerGpu 2
if ($LASTEXITCODE -ne 0) { throw "Dry-run lỗi; dừng tại đây." }

powershell -NoProfile -ExecutionPolicy Bypass -File "$S" -Mode Preflight -JobsPerGpu 2
if ($LASTEXITCODE -ne 0) { throw "Preflight lỗi; chưa chạy GPU." }
```

Preflight kiểm full input fingerprint, target/poison và CPU B32 step; không train
full. Khi có `PREFLIGHT_PASS` và GPU đủ trống, chạy capacity test riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$S" -Mode GpuSmoke -JobsPerGpu 2
```

Chỉ khi hiện `GPU_SMOKE_PASS`, launch riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$S" -Mode Launch -JobsPerGpu 2
```

Receipt v2 không được dùng thay receipt LR control. Không tự retry, giảm batch,
đổi AMP hoặc dừng process khác. Task chạy nền kiểu Interactive: có thể ngắt SSH,
giữ Vân đăng nhập và không sleep/log off/tắt máy. Helper không ghi đè task đã có;
nếu run lỗi, đọc log trước khi chủ động resume cùng source/config/backend.

## Log, kiểm tra hoàn tất và lấy kết quả

```powershell
$P = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_body_axis_s42_lr001_v3"
Get-ScheduledTask -TaskName PiW-HPELi-LRControl-Seed42 | Select-Object TaskName,State
foreach ($role in @('clean','a1')) {
    $L = "$P\hpeli\seed42\$role\console.log"
    if (Test-Path -LiteralPath $L) {
        Write-Host "===== $role ====="
        Get-Content -LiteralPath $L -Encoding UTF8 -Tail 8
    }
}
```

Controller: `$P.task.*.stdout.log`/stderr. Evaluation: `$P\evaluation.log`.
Status helper đặt đúng threading env; không cần import model/GPU/inference:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$S" -Mode Status -JobsPerGpu 2
Get-Content "$P\piw_lr_control_pilot.md" -Encoding UTF8
Import-Csv "$P\a1_six_doses.csv" |
    Format-Table dose,t_mpjpe_mm,no_trigger_t_mpjpe_mm,target_pck_pct,no_trigger_target_pck_pct,dce_mm,still_dce_mm -AutoSize
```

Sau khi **cả v2 và LR control đều hoàn tất**, comparator xác minh hai run và
in bảng đối chiếu. Đặt threading env trong chính PowerShell hiện tại: env của
helper chạy trong process con không truyền ngược về parent.

```powershell
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -B -u `
    "$env:USERPROFILE\code\ATKBackd\compare_piw_lr_control.py" `
    --v2-outdir "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_body_axis_s42_v2" `
    --control-outdir "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_body_axis_s42_lr001_v3"
if ($LASTEXITCODE -ne 0) { throw "Đối chiếu chưa đạt; đọc lỗi, không chạy lại training." }
```

Comparator chỉ đọc và replay saved artifacts; không dùng GPU, inference hoặc
training. Mismatch source/runtime, input/target/poison hoặc run chưa hoàn tất
không được bỏ qua để ghép hai bảng.

Đánh giá đồng thời Clean MPJPE/PA, same-victim target gain, DCE so với stationary,
Clean-trained+trigger và nontarget drift. Nếu LR.001 cải thiện, chỉ diễn giải kết
quả của run seed42 đã kiểm soát; không suy thành quy luật cho mọi seed/dataset.
Nếu không cải thiện, giữ kết quả và xem lại TRAIN diagnostics, không tiếp tục
quét target/threshold để chọn điểm TEST cao.

## Validation boundary

Local tests kiểm scientific-config parity, poison/target parity, checkpoint
resume, saved-output replay và gate rejection. Full processed Drive data,
concurrent CUDA capacity và hiệu quả học target phải xác nhận trên Vân.
