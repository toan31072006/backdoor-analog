# T5 v2: FT, ANP, NAD, FT-SAM trên A1, ba seed

Batch này chỉ repair checkpoint **Proposed A1/rho=.1**, MM-Fi protocol1-s1,
seeds **42/43/44** đã train 50 epochs trong T1–T3. Không train lại victim,
không đổi trigger, không thêm attack baseline. Mỗi seed có No defense + bốn
defense: tổng **15 evaluation records, 12 repair pipelines** nếu chưa reuse.
Đây không phải 12 lần train victim mới.

Bộ defense đã được điều chỉnh **sau khi xem pilot seed 42**, không phải lựa
chọn trước khi nhìn kết quả. Giữ nguyên và ghi nhận kết quả FT/Fine-Pruning
pilot; không xóa hoặc giấu kết quả FP. FT ở lại làm clean-finetuning control.
V2 không chạy thêm FP, không sửa manifest/checkpoint/cache T5 v1 hay T1–T3.

## Cấu hình và giới hạn

- FT/NAD/FT-SAM giữ nguyên recipe source-defaults đã khóa ở v1: batch 256,
  SGD lr .01, momentum .9, weight decay .0005; FT/FT-SAM 100 epochs,
  NAD teacher 10 + student 10, betas [500,1000,1000]; SAM radius 2/adaptive.
  Giữ final checkpoint, không dùng clean-best fallback.
- Chung tối đa 5% original clean TRAIN/seed, tách khoảng 80/20 **theo sequence**
  thành repair-fit/repair-selection. Original victim có thể đã thấy bank này;
  selection chỉ held out khỏi repair fitting. Không dùng TEST/attack score
  để fit hoặc chọn defense. DataLoader **workers=0**.
- ANP là **HPE-adapted**, không phải cải tiến defense hoặc reproduction CIFAR
  nguyên bản. Mask learning: 2000 iterations, batch 256, SGD lr .01/momentum .9,
  eps .4, một noise-ascent step, alpha .2; mask clamp [0,1]. Pruning thresholds
  0:.05:.9; chọn ngưỡng lớn nhất vẫn giữ clean-selection MPJPE không vượt 110%
  baseline. Nếu không ngưỡng nào đạt, giữ identity/unpruned fallback và ghi rõ.
  Loss chuyển thành MPJPE, thao tác trên BatchNorm HPELi; selection chỉ dùng
  clean TRAIN thay vì TEST/ASR. Đây là thay đổi task/protocol phải công bố.
- ANP độc lập implementation theo cơ chế nguồn, không vendor code tác giả;
  xem pin/provenance và ghi chú license tại `defend/a1_anp_sources.json`.
  Không suy license của BackdoorBench bao phủ repo ANP gốc.
- Code/local CPU tests không xác nhận cấu hình HPE bằng full MM-Fi/GPU run.
  Chưa có số liệu ANP hay kết luận kháng defense chỉ vì code đã sẵn sàng.
  Hai job/GPU không bảo đảm nhanh hơn hoặc đủ RAM/VRAM.
- Giữ native repair budget của từng phương pháp, không gọi đây là so sánh
  compute-matched. Kiểm thử resume exact trên CPU không xác nhận bitwise
  reproducibility của mọi CUDA kernel/hardware; không đổi recipe v1 để che
  khác biệt nền tảng.

## Kiểm thử local ngày 2026-10-20

**308 tests theo nhóm đã thoát exit code 0**: 60 ANP, 137 metric/runner/reuse/
Windows helper/worker, 111 regression v1. Gồm forward và một iteration ANP
trên đúng kiến trúc HPELi với tensor tổng hợp, kiểm tra RNG/sampler resume,
và cache identity fixture tổng hợp 33264 hàng. Các file scientific v1 và helper
v1 không thay đổi. Không có dữ liệu MM-Fi thật hoặc GPU full run trong con số này.

Một lượt gộp lớn bị kẹt trong môi trường Anaconda local và không được tính
là pass. Chạy riêng các nhóm với MKL/OMP một thread; nhóm reporting/launcher
cuối dùng `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` để tách plugin môi trường. Các bài
test Windows chỉ parse/mock/register giả, không tạo task thật. Cần kiểm tra log
và artifact của lần chạy MM-Fi trên Vân trước khi kết luận effectiveness.

## Chạy nền trên PowerShell máy Vân

Không chạy các lệnh này ở máy Admin rồi cho rằng job đã chạy trên Vân. Sau khi
code đã test/push, dùng **Anaconda PowerShell trên Vân**:

```powershell
Set-Location "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'git pull failed' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1DefenseTaskV2.ps1 -DryRun
if ($LASTEXITCODE -ne 0) { throw 'V2 dry-run failed' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1DefenseTaskV2.ps1 -JobsPerGpu 2
```

Mặc định chạy đủ 42/43/44, output mới:
`runs/mmfi_a1_defenses_ft_anp_nad_sam_s42_43_44_v2`.
Helper dùng `runs/mmfi_a1_defenses_s42_43_44_v1` làm nguồn reuse nếu thư mục có
sẵn; có thể chỉ định `-ReuseOutdir 'D:\path\to\v1'`, hoặc `-ReuseOutdir ''`
cho kế hoạch mới không reuse. Đường dẫn/recipe được khóa trong manifest v2;
giữ nguyên các nguồn khi resume, không đổi reuse source giữa chừng.
Để V1 hoàn tất/dừng trước khi reuse; không chạy V2 trong lúc V1 còn ghi vào
checkpoint/kết quả nguồn. Helper V2 không tự dừng task V1 thay người dùng.

Reuse chỉ sau khi audit source/checkpoint/inputs/clean-bank/recipe/NPZ/binding.
Kết quả phù hợp none/FT/NAD/FT-SAM được chép vào v2 với provenance gốc và metric
bổ sung tính từ NPZ; không mở GPU để repair/evaluate lại các record hợp lệ.
ANP là nhánh mới. Nếu artifact không khớp, runner báo lỗi thay vì gán hash v2
cho record cũ. Con số terminal pilot không đủ làm reuse artifact.

Task tên `MMFI-A1-Defense-V2`, dùng S4U của tài khoản hiện tại, không lưu mật
khẩu; chỉ đường dẫn local không EFS/network. Đóng SSH/VS Code không dừng task.
Sleep/shutdown/OOM/process bị kill vẫn làm gián đoạn. Không auto-retry hoặc
auto-start sau reboot, không tự giảm batch. Không có giới hạn 72 giờ mặc định.

Muốn **resume sau khi task/process đã dừng**, gọi lại cùng helper. Nếu v2 cũ
vẫn đang chạy và muốn dừng đúng nó để resume, dùng tường minh:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1DefenseTaskV2.ps1 -JobsPerGpu 2 -StopExisting
```

Preflight read-only phải đạt trước khi stop/register task. `-StopExisting` chỉ
nhắm **V2 đúng output và cùng user**, không dừng V1/YOLO/T1–T3. Không xóa lock,
checkpoint hay kết quả để ép resume. Epoch/ANP iterations chưa checkpoint có
thể lặp lại; checkpoint epoch/stage đã lưu được giữ.

## Xem tiến trình và kết quả

```powershell
$DOSE_V2 = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_defenses_ft_anp_nad_sam_s42_43_44_v2"
Get-ScheduledTask -TaskName MMFI-A1-Defense-V2 | Select-Object TaskName,State
Get-ChildItem "$DOSE_V2.task.*.stdout.log" | Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Encoding UTF8 -Tail 20 }
Get-ChildItem "$DOSE_V2.task.*.stderr.log" | Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Encoding UTF8 -Tail 20 }
Get-Content "$DOSE_V2\seed42\a1\anp\console.log" -Encoding UTF8 -Tail 15 -Wait
```

`Running` không chứng minh tiến trình đang tăng; xem log iteration/epoch và
thời gian sửa file. Ctrl+C chỉ dừng lệnh `Get-Content -Wait`, không dừng task.
Log stdout/stderr có timestamp + suffix riêng để không tranh file startup cũ.
Preflight full-identity/bank audit có thể cần thời gian trước dòng log cell.

Sau khi controller đã kết thúc, xuất/audit lại (không repair/model-evaluate):

```powershell
$DOSE_SOURCE = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1"
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_a1_defenses_v2.py --source-outdir "$DOSE_SOURCE" `
  --outdir "$DOSE_V2" --export-only
if ($LASTEXITCODE -ne 0) { throw 'V2 export/audit failed' }
Get-Content "$DOSE_V2\table5_defenses.md" -Encoding UTF8
```

Không mở exporter cùng lúc với controller giữ root lock. `--status` chỉ đếm
artifact presence, **không phải fresh scientific audit**. Chỉ đủ 15 record đã
audit mới báo bảng ba seed complete; seed thiếu vẫn là partial, không bịa SD.

## Metric: đánh giá defense, không chỉ làm target score giảm

| Metric chính | Scope, dose 1, không alignment | Hướng defender |
| --- | --- | --- |
| T-MPJPE | Prediction triggered so với attacker target, joints [2,3] | tăng |
| Target-PCK | Joint-level %, threshold .2 × clean GT scale (5,12), joints [2,3] | giảm |
| Clean MPJPE | Input không trigger so với GT thật, toàn bộ 17 khớp | giảm |
| Triggered-GT MPJPE | Prediction triggered so với GT thật, chỉ [2,3] | giảm |

Clean MPJPE 17 khớp **không cùng scope** với Triggered-GT hai khớp. CSV/JSON v2
thêm clean-target MPJPE [2,3] của **chính model ấy**, cùng-joint recovery gap
`Triggered-GT − clean-target`, Clean PA-MPJPE, untriggered T-MPJPE/Target-PCK
đối với target d=1, và đầy đủ sáu dose/DCE. Gap âm hợp lệ, không clamp về zero.
Đây không phải clean-trained-model recovery floor và không chứa metric 17-joint
triggered-GT chưa được lưu. Target-PCK không phải pose-level ASR.

Mỗi metric lấy mean ± **sample SD (ddof=1)** trên ba seed độc lập, không trên
frame. Xuất thêm **repair − No defense ghép cặp trong từng seed**, rồi mới
tổng hợp mean/SD; chênh lệch PCK tính theo percentage points. Không lấy hiệu hai
SD. DCE là diagnostic của attacker, **không phải càng giảm thì defense càng tốt**.
Target-PCK thấp một mình không chứng minh model lấy lại pose thật; phải xem
Triggered-GT, same-joint clean reference và utility loss cùng lúc.

Artifacts chính: `table5_defenses.md`, `defense_summary.json`,
`defense_per_seed.csv`, `defense_aggregates.csv`, `defense_paired_deltas.csv`,
`defense_paired_aggregates.csv`, `defense_six_dose.csv`; mỗi cell giữ NPZ,
evaluation checksum/provenance và repair checkpoint nếu có.
