# T5 local: FT, Fine-Pruning, NAD và FT-SAM cho MM-Fi

## Cập nhật 2026-10-19: pilot seed 42, tối đa hai job trên một GPU

Người dùng báo đã chạy xong T1–T3 với seeds 42/43/44 trên máy Vân và yêu cầu
chạy T5 seed 42 với **2 job song song**. Đây là thay đổi runtime, không thay
defense/config/metric, không phải full-run T5 đã hoàn tất. Các mục tuần tự bên
dưới là hướng dẫn runner gốc và được giữ để truy vết.

Sidecar `ATKBackd/run_a1_defenses_parallel.py` gọi **chính worker gốc** của
`run_a1_defenses.py`; không sửa bốn file đã khóa trong T5 `sources`. Vì vậy
manifest T5, `repair.pt` và kết quả có binding hợp lệ từ lượt tuần tự vẫn dùng
được. Không sửa hash hoặc tạo profile mới chỉ để đổi số slot.

Kiểm thử local: 25 test mới đạt (22 scheduler/lock/provenance/toy child-process,
3 PowerShell parse/ownership/dry-run), cùng 246 test hồi quy defense/T1–T3 đã
đạt. Không đăng ký task hay chạy full MM-Fi/GPU trong các test này. Lượt test
riêng 25 case cuối chạy trên phiên bản code đã ổn định; warning pin-memory
thuộc các fixture CPU cũ, không phải số liệu benchmark.

- Mặc định seed 42, đủ `none`, `ft`, `fp`, `nad`, `ft_sam`; tối đa 2 child trên
  `cuda:0`, DataLoader vẫn workers=0. `--jobs-per-gpu 1` giữ một slot; không nhận
  3 slot. Số slot không làm batch 256 thành 128 hoặc thay số epochs.
- Controller giữ cùng khóa root với runner gốc, kiểm tra cả cell không được chọn
  để từ chối orphan worker. Không xóa `run.lock`; không mở controller thứ hai.
- Audit source và tạo/kiểm tra clean bank chung **trước khi** spawn các child,
  tránh hai worker cùng ghi `clean_bank.json.tmp`. Dataset của preflight được
  giải phóng trước khi chạy hai child; mỗi child vẫn tự audit source/bank.
- Kết quả hoàn tất chỉ skip sau audit source/input/bank/NPZ/checkpoint/binding.
  Repair chưa xong resume tại checkpoint epoch/stage đã lưu; log nối tiếp.
  Runtime provenance nằm riêng tại `runtime_launches/`, không sửa scientific plan.
- Child lỗi thì controller dừng, không retry/tự giảm batch/đổi defense. Controller
  chỉ cleanup worker do chính nó spawn; không tìm và kill Python/YOLO khác.
  Khi chuyển launcher, epoch chưa lưu có thể phải chạy lại.
- Chưa đo peak VRAM/RAM hoặc speedup khi hai defense cùng chạy trên RTX 4090.
  Đặc biệt NAD giữ teacher và student; thiếu bộ nhớ thì báo lỗi, không coi 24 GB
  là bảo đảm. Đây vẫn là pilot một seed, không phải bảng T5 ba seed.

### PowerShell trên máy Vân: chạy nền, có thể ngắt SSH

Sau khi code đã được kiểm thử và push, chạy trong **PowerShell máy Vân**:

```powershell
Set-Location "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'git pull failed' }

powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-A1DefenseTask.ps1 `
  -Seeds 42 -JobsPerGpu 2 -StopExisting
```

`-StopExisting` là yêu cầu tường minh: dừng task/worker **T5 gắn đúng output**
`runs/mmfi_a1_defenses_s42_43_44_v1` trước khi chạy controller mới; giữ mọi
checkpoint/kết quả, không chạm job T1–T3, YOLO hoặc T5 output khác. Script dry-run
kiểm tra plan trước khi dừng. Bỏ cờ này thì script từ chối nếu T5 cũ còn sống.
Task mới tên `MMFI-A1-Defense-Parallel`; script nhận diện cả task tuần tự cũ
`MMFI-A1-Defense-S42`. Không start task seed-44 để chạy defense.

Helper dùng full Conda executable đã có trong `CONDA_EXE` và `conda run` để
activate môi trường; không lưu mật khẩu. Task S4U chỉ dùng đường dẫn local,
không hỗ trợ tài nguyên mạng/EFS. Không giới hạn mặc định 72 giờ, không auto-retry
hay auto-run sau reboot. Task/permission/CUDA vẫn phải xác minh trên máy Vân.
Đóng SSH/VS Code không phải lệnh dừng Task; sleep, shutdown, OOM vẫn làm gián đoạn.

Chỉ kiểm tra trước, không tạo task/stop job/load dataset hoặc CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote_tools\Start-A1DefenseTask.ps1 -DryRun
```

Kiểm tra task và log (Running không tự chứng minh epoch đang tăng):

```powershell
Get-ScheduledTask -TaskName MMFI-A1-Defense-Parallel | Select-Object TaskName,State
$DOSE_DEFENSE = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_defenses_s42_43_44_v1"
Get-Content "$DOSE_DEFENSE.task.*.stdout.log" -Encoding UTF8 -Tail 12
Get-Content "$DOSE_DEFENSE.task.*.stderr.log" -Encoding UTF8 -Tail 12
Get-Content "$DOSE_DEFENSE\seed42\a1\*\console.log" -Encoding UTF8 -Tail 8
```

Preflight audit/bank có thể chưa in log cell ngay. Đọc log controller và process
trước khi kết luận treo. Pilot đủ 5/15 records vẫn báo partial như runner gốc.
Muốn chạy foreground (vẫn phụ thuộc terminal):

```powershell
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_a1_defenses_parallel.py `
  --source-outdir "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1" `
  --outdir "$DOSE_DEFENSE" --device cuda:0 --jobs-per-gpu 2 --seeds 42
```

## Phạm vi hiện tại

Ngày 2026-10-18 người dùng đồng ý **tạo code, kiểm thử và push GitHub**.
Chỉ đạo mới nhất: kiểm tra seed 42/43 của T1–T3 trước, defense để chạy sau.
File này không có nghĩa đã khởi chạy defense hoặc đã có kết quả T5 trên máy Vân.
Không thay đổi batch T1–T3, checkpoint, trigger hay config các phương pháp.

Theo chỉ đạo mới nhất, **chỉ chạy defense trên Proposed A1**, chỉ adapt để chạy
CSI/HPE, không cải tiến defense, không chạy lại baseline truyền thống. Đây là
**source-defaults, task-adapted**, không phải nguyên pipeline phân loại của
BackdoorBench hoặc setting đã tối ưu cho MM-Fi. Profile:
`mmfi_a1_defenses_source_default_adapt_v1`. Chưa có full-run validation.

Nguồn đã pin: BackdoorBench commit
`f02e3534645f0ee63d6848653062cd6c0d6c400d`.
`ATKBackd/defend/a1_defense_sources.json` lưu URL, SHA256 và byte length của FT,
FP, NAD, FT-SAM, bốn default YAML, optimizer/SAM helper cùng license. Đây là provenance, không vendor
nguyên repo. License gốc khai báo CC-BY-NC-4.0 và có ngoại lệ third-party;
không suy tất cả thành phần có cùng license, không tuyên bố tác giả xác nhận bản adapt.

## 0. Bước hiện tại: kiểm tra seed 42/43, chưa chạy defense

Sau khi pull code trên máy Vân, lệnh sau chỉ đọc artifact của T1–T3. Không train,
không đánh giá model mới, không tạo/sửa checkpoint hay export bảng:

```powershell
$env:PYTHONUTF8 = "1"
$DOSE_OUT = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1"
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\check_a1_three_tables.py --outdir "$DOSE_OUT" --seeds 42 43
```

Chỉ coi cả hai seed đủ khi từng seed có **VERIFIED 13/13** và cuối log là
`ALL_SELECTED_SEEDS_VERIFIED`. `MISSING` là thiếu artifact; `INVALID` là kiểm tra
binding/config/checkpoint/input/poison/NPZ không đạt. Gửi log để xác định cell
cần xử lý trước khi chọn seed resume; không tự chuyển sang seed 44 hoặc defense.
Audit cần bộ dữ liệu gốc để kiểm tra ordered identities, có thể mất vài phút.

## 1. Ma trận cần chạy

- Nguồn: root `three_tables.resolved.json` đã chạy T1–T3; không lấy checkpoint
  15 epoch hay các run cũ khác config.
- Từng seed chỉ gồm **Proposed A1 tại rho=.1**, đọc từ checkpoint T1–T3 đã có.
  Không repair thêm clean model, không chạy defense trên baseline truyền thống.
- **5 setting**: `none` (No defense), `ft`, `fp`, `nad`, `ft_sam`.
- Mặc định chọn **seed 42 để pilot**; seeds 43/44 chỉ chọn khi checkpoint nguồn
  đã complete và hợp lệ. Không cần chờ các cell rho=.2/.4 hay baseline nếu A1
  đã complete. Không tự train bù một victim thiếu checkpoint.
- Đủ một seed có **5 hàng đánh giá**, gồm 1 No-defense và 4 repair pipelines.
  Ba seed có 15 hàng, 12 repair pipelines. NAD có cả teacher và student nên
  số lượt tối ưu thực tế không bằng số hàng kết quả.
- Victim nguồn vẫn là full MM-Fi protocol1-s1/split seed 0, 50 epochs,
  rho=.1 cho A1. Defense fine-tuning không phải retrain victim từ đầu.
- Mọi output/checkpoint defense đặt ở **root mới tách biệt**, không bên trong
  source run; không ghi đè checkpoint, cache hoặc manifest T1–T3.

**Clean MPJPE của chính A1 trước/sau defense vẫn bắt buộc**; nó đo utility trên
input sạch, không phải thêm một clean-trained model. Không tuyên bố kết quả này
đã đánh giá defense trên mọi attack hoặc đã có clean-model-repair control.

## 2. Trusted-clean bank và tránh TEST leakage

Mỗi seed dùng một bank chung từ **original clean TRAIN**, tối đa 5% số frame.
Bank được chia khoảng 80/20 thành **repair-fit** và **repair-selection**;
hai phần không trùng sequence. Lưu IDs, group IDs, số frame thực tế và hash.
Cùng seed phải có cùng bank cho bốn defense trên A1. Source dùng image/class
subset handling; ở đây dùng common sequence-balanced bank, khai báo rõ đây là
adaptation dữ liệu chứ không phải giữ nguyên class sampler.

- Fit model, xếp activation FP và fit teacher chỉ dùng repair-fit.
- Chỉ mức prune/FP stopping dùng clean MPJPE trên repair-selection. Mọi defense
  dùng **model cuối lịch repair**, không chọn best epoch hoặc fallback model gốc.
- Không dùng trigger, attacker target, attack success hoặc official TEST để
  chọn tham số. `validation` trong pipeline MM-Fi cũ trùng official TEST,
  **không được coi đó là defense validation**.
- Original victim có thể đã thấy cả fit/selection này trong original TRAIN;
  selection chỉ held out khỏi **repair fitting**, không phải dữ liệu chưa từng
  được victim nhìn thấy. Trusted-clean assumption gồm CSI sạch và pose thật.
- Khóa checkpoint defense trước khi mở full TEST (33264 frame). Không cap 512
  frame, không tự downsample TEST và không thay clean PCK thành Target-PCK.

## 3. Cơ chế và default từ source

| Setting | Giữ cơ chế nào? | Khai báo adaptation HPE |
| --- | --- | --- |
| No defense | Load đúng checkpoint gốc đã audit | Không tối ưu trọng số |
| FT | Fine-tuning bằng clean data | MPJPE thay cross-entropy |
| FP | Xếp clean activation, prune rồi fine-tune | Mask cột input của final linear `regression.7`; không prune class logits |
| NAD | Teacher fine-tune, frozen teacher, attention-transfer student | Attention ở `skunit1`, `skunit2`, `regression.5`; không dùng output-only distillation |
| FT-SAM | Weight-adaptive, hai lượt forward/backward | MPJPE, fixed radius; không bê class sampling/label smoothing |

Giữ default số của pinned BackdoorBench: SGD lr=.01, momentum=.9,
weight decay=.0005, batch size 256, clean ratio=.05. **FT, FP, FT-SAM: 100
repair epochs**. FT/FP dùng CosineAnnealingLR T_max=100, step mỗi epoch.
NAD source YAML có `epochs=100` nhưng code thực tế dùng `te_epochs=10` cho cả
teacher lẫn student: **10+10 epochs**, không phải student 100 epochs.

FP xếp **mean activation có dấu**; step count `ceil(.01 * N_features)`, bắt đầu
từ 0 và dừng khi lần đầu vi phạm utility gate. Source tolerance `.1` trên
relative clean-accuracy change được adapt thành **relative clean-selection
MPJPE increase ≤10%**. Source dùng clean TEST để chọn pruning; bản này bắt buộc
chuyển sang clean TRAIN-selection. Sửa off-by-one slice và alias candidate của
source, khai báo rõ, không thêm loss hoặc tối ưu attack-aware. Giữ curve/mask
và lý do dừng; mask phải còn tác dụng sau fine-tuning/save/load.
FP ở đây bỏ các **input features của final linear head** bằng mask cột weight,
không tuyên bố prune toàn bộ convolutional channel của backbone.

NAD giữ attention power p=2, betas **[500,1000,1000]** cho ba layer theo thứ tự
trong bảng; normalization chia norm cộng 1e-6. Teacher giữ cosine T_max=100
trong 10 epoch; student dùng manual LR source (.01 trong 10 epoch mặc định).
Teacher cuối được frozen; student bắt đầu từ A1 gốc, không từ teacher đã fine-tune.

FT-SAM giữ radius **2.0**, adaptive=True, alpha=0 như source; source có
rho_min=rho_max=2 nên radius không đổi. Cosine T_max=100 **step mỗi minibatch**
đúng operative code, không âm thầm thay thành epoch schedule. Không dùng
label smoothing hoặc artificial class sampling vì pose là regression; MPJPE
thay classification loss. Không thêm module hay sửa defense để A1 khó bị xóa hơn.

Mọi nhánh giữ **checkpoint cuối**, không clean-best/epoch -1 fallback. Nếu
default source làm clean utility hỏng, báo đúng và chưa gọi đó là defense thành
công; muốn đổi setting phải tạo profile mới, dựa clean-only validation và khai
báo, không âm thầm làm yếu defense sau khi nhìn attack TEST.

**Windows:** T5 dùng DataLoader `num_workers=0` và chạy lần lượt để tránh lặp lại
lỗi shared-memory 1455. Đây là execution policy của T5 mới; không sửa worker
policy hay scientific hash của T1–T3. Không bảo đảm tốc độ/RAM trước khi đo thực tế.

## 4. Metric cần đọc đúng chiều

| Cột chính | Định nghĩa | Chiều tốt cho defender |
| --- | --- | --- |
| T-MPJPE (mm) | Triggered prediction so với attacker target, joints [2,3], không align | Tăng |
| Target-PCK (%) | Khớp target tại .2 × clean-GT distance joints (5,12), joints [2,3] | Giảm |
| Clean MPJPE (mm) | Clean prediction so với pose thật, toàn bộ 17 khớp | Giữ thấp |
| Triggered GT-MPJPE (mm) | Triggered prediction so với **pose thật**, joints [2,3] | Giảm |

Cột target chính tính d=1; định nghĩa dose/ngưỡng ở caption, không cần thêm
`@1` vào header. T-MPJPE tăng nhưng GT-MPJPE cũng tăng có thể chỉ là model hỏng,
không phải defense thành công. Không gọi Target-PCK là pose-level ASR.

Giữ sáu dose d=0,.2,.4,.6,.8,1 và DCE trong artifact. DCE thấp thể hiện model
làm theo attacker dose tốt hơn, **không phải defense tốt hơn**. Không tự tô đậm
DCE thấp như thành công phòng vệ. Clean PA-MPJPE, no-trigger target controls
và per-sample arrays là diagnostics, không thay bốn metric chính.

Một seed chỉ là pilot. Chỉ sau đủ seeds 42/43/44 mới báo mean ± sample SD
(ddof=1), không lấy frame liền kề làm seed độc lập và không chọn seed tốt nhất.

## 5. PowerShell trên máy Vân — chỉ dùng sau khi đã nhận code mới

Các lệnh dưới đây là hướng dẫn cho bước tiếp theo, **không được thực thi bởi
việc đọc tài liệu này**. Code được phép push sau test; defense hiện để sau. Không chạy
T5 song song batch T1–T3 đang thiếu RAM; tránh đổi scientific source khi launcher
cũ còn sống. Không dừng YOLO hoặc process người khác.

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
Set-Location "$env:USERPROFILE\code"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DOSE_SOURCE = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1"
$DOSE_DEFENSE = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_defenses_s42_43_44_v1"

python -u .\ATKBackd\run_a1_defenses.py `
  --source-outdir "$DOSE_SOURCE" --outdir "$DOSE_DEFENSE" `
  --device cuda:0 --seeds 42 --dry-run
```

Dry-run là **read-only**, không tạo output/plan/checkpoint, không load data/GPU
hay sửa source. Nó không chứng minh checkpoint/dataset đã đủ hay repair hiệu quả.

Khi có quyền chạy pilot, bỏ `--dry-run`:

```powershell
python -u .\ATKBackd\run_a1_defenses.py `
  --source-outdir "$DOSE_SOURCE" --outdir "$DOSE_DEFENSE" `
  --device cuda:0 --seeds 42
```

Mặc định chỉ A1 và đủ 5 setting. Khi cần xem hoặc tiếp tục:

```powershell
python -u .\ATKBackd\run_a1_defenses.py `
  --source-outdir "$DOSE_SOURCE" --outdir "$DOSE_DEFENSE" --seeds 42 --status

python -u .\ATKBackd\run_a1_defenses.py `
  --source-outdir "$DOSE_SOURCE" --outdir "$DOSE_DEFENSE" `
  --device cuda:0 --seeds 43 44
```

Một plan T5 ổn định hỗ trợ chạy seed 42 trước rồi 43/44 cùng root. Chỉ reuse
artifact đã qua kiểm tra provenance; không tự chép completion JSON hoặc sửa hash.
Nếu source chưa complete/khác recipe, dừng thay vì tự train hay nhận cache khác.
Không mở hai controller cùng output T5.

Output gồm `seed42/a1/<defense>/console.log`, `repair.pt` cho repair,
`evaluation.json`, `per_sample.npz`; root giữ `defense_per_seed.csv`,
`defense_six_dose.csv`, `defense_summary.json` và `table5_defenses.md`.
Sau pilot 42 đủ **5/15 records**, nhãn `T5_PARTIAL_PREVIEW...` là đúng vì còn
43/44, không phải pilot bị thiếu một defense. `--status` chỉ đếm artifact presence,
không thay cho source/record audit. Chỉ sau đủ 15 records hợp lệ mới có bảng ba seed.

Nếu cần xuất lại các record đã hoàn tất sau một lần dừng, khi **không còn controller
T5 chạy** có thể dùng `--export-only` với cùng source/outdir. Nó không repair hoặc
inference lại, nhưng vẫn audit checkpoint/input identities và ghi report; không
được dùng để bỏ qua record stale hay biến preview thành bảng hoàn chỉnh.

## 6. Giới hạn và kiểm thử

Kiểm thử local ngày 2026-10-18: **334 tests passed**, gồm 86 test mới cho T5,
22 test checker read-only và test hồi quy T1–T3/Target-PCK/DCE. Chạy CPU bằng
fixture tổng hợp; ba warning
pin-memory thuộc test pipeline cũ khi không có accelerator. Chưa chạy full MM-Fi
hay benchmark defense trên máy Vân. Quyền push code không phải kết quả thí nghiệm.

Test local phải kiểm tra công thức metric, full-length evaluation, trusted-bank
sequence disjointness, no-TEST selection, FP mask persistence, NAD teacher/student,
FT-SAM restore/update, checkpoint/provenance corruption và dry-run không ghi file.
Toy CPU tests không phải full MM-Fi/GPU validation; không điền số giả vào paper.
Thời gian chạy, mức RAM/GPU và hiệu quả defense chỉ được báo sau đo thật.

Bốn phương pháp này là bộ model-repair đại diện. Chưa triển khai NC/STRIP,
ANP/RNP, CSI preprocessing, detector hay bảng T4/T6/T7 trong lượt này. Không
suy ra kháng mọi defense hoặc kháng RF/over-the-air từ kết quả tensor CSI.

Nguồn cơ chế đã kiểm tra: [FT](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/defense/ft.py),
[FP](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/defense/fp.py),
[NAD](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/defense/nad.py),
[FT-SAM](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/defense/ft-sam.py).
