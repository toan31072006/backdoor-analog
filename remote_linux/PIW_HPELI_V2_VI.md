# PiW v2: HPE-Li Clean + A1, seed42

Exploratory development pilot, chưa phải full comparison/bốn bảng paper. V1 đã
được xem TEST nên v2 không phải blind test. Giữ nguyên output/source v1; không
resume hoặc đổi target của checkpoint cũ rồi gọi đó là model đã học target v2.

## Cấu hình đóng băng trước lượt chạy mới

| Thành phần | V1 | V2 |
| --- | --- | --- |
| Victim | HPE-Li hiện có | Giữ nguyên architecture, corrected SK attention |
| Scratch training | AdamW lr .001, drop_last=False | AdamW lr .01, drop_last=True |
| Lịch | 200 epoch, final199, B32 | Giữ nguyên |
| Target chân phải | Global Z40° × dose | 40° × dose quanh normalize(GT5 − GT7), pivot7, joints11/13 |
| PCK | .2 × khoảng cách clean GT6/4 | Giữ nguyên, không nới ngưỡng |
| Trigger | .185, amplitude only, rho .1 | Giữ nguyên operator, indices và paired doses |
| Clean reference | Historical v1 | Train Clean v2 mới, cùng recipe với A1 v2 |

FP32, workers0, ordinary MPJPE, fixed LR, AdamW wd .01/betas(.9,.999)/eps1e-8.
Không scheduler, AMP, pretrained weights, early stopping hoặc chọn epoch theo
TEST. TRAIN shuffle/drop_last dùng 27872/27876 mẫu mỗi epoch (đuôi ngẫu nhiên theo
shuffle), TEST dùng đủ 2586 mẫu. Split giữ official one-person và next-frame TRAIN
filter; không phải held-out-session protocol. Cả hai model train độc lập; evaluation
chờ cả hai hoàn tất final199, không dùng Clean để warm-start A1.

Nguồn recipe: [DT-Pose scratch HPE-Li branch](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L125),
[PiW config](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/config/person_in_wifi_3d/pose_config.yaml),
[loader/drop_last](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/feeder/person_in_wifi_3d.py).
Đây là recipe adaptation; không claim bit-identical author reproduction hoặc proposed DT-Pose model.

Target xoay rigid nhánh chân quanh trục ngang hai hông. Không mặc định tên trục
camera là chiều cao thế giới. Hip axis ≤1e-6 m/nonfinite làm run dừng, không tự
đổi trục/bỏ frame. Giữ bone lengths nhưng chưa chứng minh mọi target đều hợp lý
giải phẫu/không xuyên sàn. 40° giữ angular budget cũ, không chọn theo TEST để lấy
điểm đẹp. Recipe và target cùng thay đổi, nên kết quả không cô lập hiệu ứng từng
thay đổi. Muốn attribution cần run kiểm soát riêng.

## Chạy trên Vân: PowerShell, không phải CMD

Code trong `C:\Users\Van\code`, data/action/env giữ ở `backdooranalog`.
Output mới `runs\piw_hpeli_body_axis_s42_v2`. Không gọi launcher v1 cho output này.

```powershell
cd "$env:USERPROFILE\code"
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-PiWHPELiV2Task.ps1 -Mode DryRun -JobsPerGpu 2
```

DryRun chỉ đọc source/package metadata, không đọc dataset, import torch hoặc dùng
GPU/ghi output. Sau đó chạy CPU/data/model preflight:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-PiWHPELiV2Task.ps1 -Mode Preflight -JobsPerGpu 2
```

Preflight hash full inputs, kiểm hip axis, tính geometry/distortion trên TRAIN,
chạy một optimizer step CPU B32 rồi ghi `...v2.preflight.json`. Đây không phải
full training. Xem `diagnostics.geometry` trước khi chạy GPU.

Khi GPU đủ trống, capacity smoke chạy đúng số process/physical B32:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-PiWHPELiV2Task.ps1 -Mode GpuSmoke -JobsPerGpu 2
```

Chỉ khi có `GPU_SMOKE_PASS`, mới launch riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-PiWHPELiV2Task.ps1 -Mode Launch -JobsPerGpu 2
```

Không tự retry, giảm batch hoặc kill job khác khi OOM. Nếu hai process không vừa,
chọn output mới và `-JobsPerGpu 1`, làm lại gates của cấu hình đó. Hai victim sẽ
train tuần tự. Gate fingerprint khóa source/config/runtime/input/device. File
receipt PASS chỉ xác nhận test đó, không bảo đảm full training/efficacy.
Task là Interactive: giữ user Vân đăng nhập; ngắt SSH được, không log off/sleep
hoặc tắt máy. Không có restart tự động. Muốn resume sau lỗi: xem log trước; runner
`--launch` resume checkpoint atomic cuối với cùng source/config/backend. Helper
không ghi đè task đã tồn tại.

## Log và kết quả

```powershell
$P = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_body_axis_s42_v2"
Get-ScheduledTask -TaskName PiW-HPELi-V2-Seed42 | Select-Object TaskName,State
foreach ($role in @('clean','a1')) {
    $L = "$P\hpeli\seed42\$role\console.log"
    if (Test-Path -LiteralPath $L) {
        Write-Host "===== $role ====="
        Get-Content -LiteralPath $L -Encoding UTF8 -Tail 8
    }
}
```

Controller log: `$P.task.*.stdout.log` và stderr. Evaluation: `$P\evaluation.log`.
Mỗi epoch lưu checkpoint, optimizer và RNG; source/runtime/input/poison binding
không cho reuse sai profile. Audit hoàn tất không chạy inference/training:

```powershell
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  "$env:USERPROFILE\code\ATKBackd\run_piw_hpeli_v2.py" --outdir "$P" --status
Get-Content "$P\piw_v2_pilot.md" -Encoding UTF8
Import-Csv "$P\a1_six_doses.csv" | Format-Table dose,t_mpjpe_mm,no_trigger_t_mpjpe_mm,target_pck_pct,no_trigger_target_pck_pct,dce_mm,still_dce_mm -AutoSize
```

Lưu đầy đủ predictions/GT/targets/IDs/sáu dose, cùng-victim no-trigger và Clean+
trigger controls. T-MPJPE/Target-PCK chấm11/13, không alignment; clean metrics
chấm14khớp. DCE summary mean5dose dương. Target-PCK không phải ASR; không so
absolute scores v1/v2 như cùng target hoặc gộp PiW/MM-Fi thành một score.

## Local validation

CPU synthetic tests kiểm geometry/rigid equivariance, paired poisons, exact
checkpoint resume, deferred evaluation và replay/tamper. Full CUDA capacity,
full processed Drive mirror và learned efficacy phải xác nhận trên máy chạy.
