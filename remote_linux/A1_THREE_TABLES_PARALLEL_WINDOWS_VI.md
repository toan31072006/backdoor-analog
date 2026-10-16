# Tiếp tục T1–T3 trên Windows: hai victim cùng một RTX 4090

## Phạm vi và điều kiện

Đây là hướng dẫn đổi lịch thực thi, **không phải một thí nghiệm mới**. Helper
`ATKBackd/run_a1_three_tables_parallel.py` chỉ tiếp tục manifest đã tồn tại;
không tạo lại plan, đổi recipe hoặc tăng số victim. Bản hướng dẫn/code local
không có nghĩa file đã được push, máy Vân đã nhận file hoặc job đã được chạy.

- Máy đích: Windows, tài khoản Vân, PowerShell; code ở `C:\Users\Van\code`.
- Môi trường: `C:\Users\Van\backdooranalog\envs\dose-backdoor`.
- Tiếp tục đúng output `C:\Users\Van\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1`.
- T1/T2/T3, seeds 42/43/44; 13 scientific victims/seed, tổng 39. Mỗi seed dùng
  chung đúng A1 giữa ba bảng; không train A1 lặp lại.
- Full MM-Fi protocol1-s1/split seed 0; 50 epochs; HPELi; SGD lr=.001,
  momentum=.9, weight decay=0; batch size 32; giữ nguyên workers trong manifest.
- Giữ đủ clean reference, BadNets, WaNet, FTrojan, FIBA, SIG, BppAttack và A1;
  không có Blended. Giữ A1–A4 và rho=.1/.2/.4, sáu evaluation doses như plan.
- T1: T-MPJPE, Target-PCK, Clean MPJPE, Clean PA-MPJPE. T2/T3: các metric đã
  chốt trong `MMFI_CVPR_EXPERIMENT_PLAN_20261015.md`, không thay định nghĩa.
- Helper mới phải được đặt cạnh runner gốc. Không sửa file scientific source,
  baseline hoặc manifest để vượt kiểm tra hash; không pull một bản thay đổi
  scientific sources khi controller cũ còn chạy.

Hai slot không bảo đảm nhanh hơn. Số quan sát trước đó khoảng 4/24 GB VRAM,
utilization khoảng 70% còn bao gồm process YOLO khác. Chưa benchmark hai slot
trên GPU thật; không tự dừng YOLO hoặc process của người khác.

## 1. Kiểm tra chỉ đọc, chưa dừng job cũ

Mở **PowerShell thứ hai** trên máy Vân, giữ cửa sổ đang train mở:

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
Set-Location "$env:USERPROFILE\code"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$DOSE_OUT = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1"

Test-Path .\ATKBackd\run_a1_three_tables_parallel.py
Test-Path "$DOSE_OUT\three_tables.resolved.json"

python -u .\ATKBackd\run_a1_three_tables_parallel.py `
  --outdir "$DOSE_OUT" --devices cuda:0 --jobs-per-gpu 2 --dry-run
```

Hai `Test-Path` phải là `True`. Nếu helper chưa có trên máy Vân, nhận đúng
file mới trước; không gọi một file chưa được chuyển tới. Dry-run của **helper
mới** chỉ đọc manifest/source contract; có thể chạy trong khi controller cũ
đang train. Không dùng dry-run của runner gốc để thao tác cùng output đang chạy.
Dry-run không kiểm tra toàn bộ dữ liệu/checkpoint hoặc xác nhận tốc độ GPU.

Xem checkpoint/log gần nhất (chỉ đọc):

```powershell
Get-ChildItem "$DOSE_OUT\seed*\*\checkpoint.pt" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 3 FullName, LastWriteTime, Length

Get-ChildItem "$DOSE_OUT\seed*\*\console.log" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 15 }
```

Đợi cell đang train ghi dòng `[ckpt] saved` xong rồi mới chuyển lịch. Đừng mở
hoặc sao chép một checkpoint trong lúc nó đang được ghi. Nếu chưa có checkpoint
của cell đó, dừng lúc này có thể khiến riêng cell đó bắt đầu lại từ epoch 0.

## 2. Ghi nhận đúng process của run rồi dừng controller cũ

Trong PowerShell thứ hai, chụp cây process trước khi dừng để còn nhận diện các
DataLoader descendants có thể sống sót sau khi parent thoát. Lọc theo đúng
tên runner **và output**, không liệt kê/kill toàn bộ Python:

```powershell
$DOSE_PROCESS_SNAPSHOT = @(Get-CimInstance Win32_Process)
$DOSE_RUN_ROOTS = @($DOSE_PROCESS_SNAPSHOT | Where-Object {
  $_.Name -match '^python(w)?\.exe$' -and
  $_.CommandLine -match 'run_a1_three_tables(?:_parallel)?\.py' -and
  $_.CommandLine -and
  $_.CommandLine.IndexOf($DOSE_OUT, [StringComparison]::OrdinalIgnoreCase) -ge 0
})
$DOSE_TRACKED_IDS = @($DOSE_RUN_ROOTS | ForEach-Object { [int]$_.ProcessId })
do {
  $DOSE_NEW_CHILDREN = @($DOSE_PROCESS_SNAPSHOT | Where-Object {
    [int]$_.ParentProcessId -in $DOSE_TRACKED_IDS -and
    [int]$_.ProcessId -notin $DOSE_TRACKED_IDS
  })
  $DOSE_TRACKED_IDS += @($DOSE_NEW_CHILDREN | ForEach-Object { [int]$_.ProcessId })
} while ($DOSE_NEW_CHILDREN.Count -gt 0)
$DOSE_PROCESS_SNAPSHOT |
  Where-Object { [int]$_.ProcessId -in $DOSE_TRACKED_IDS } |
  Select-Object ProcessId, ParentProcessId, Name, CommandLine |
  Format-List
```

Nếu không tìm thấy đúng controller/cell, chưa chuyển tiếp: kiểm tra lại output
và command line. Danh sách có thể gồm controller, worker và DataLoader, không
phải mỗi PID là một victim train riêng.

Sau đó quay lại **cửa sổ chạy training cũ** và nhấn **Ctrl+C một lần**, chờ
controller thoát và prompt trở lại. Đây là chủ động dừng training, không phải
Ctrl+C trong cửa sổ chỉ xem log.

Ở PowerShell thứ hai kiểm tra lại các PID đã chụp:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { [int]$_.ProcessId -in $DOSE_TRACKED_IDS } |
  Select-Object ProcessId, ParentProcessId, Name, CommandLine |
  Format-List
```

Chỉ tiếp tục khi controller, cell và DataLoader descendants của run cũ đã
thoát. Nếu còn PID trong danh sách hoặc vẫn thấy job ghi log, **không mở
controller mới**; gửi cây process/log để xác minh. Không dùng
`Stop-Process -Name python`, không dừng toàn bộ process trong `nvidia-smi`,
không xóa `run.lock` hoặc đổi output để lách khóa.

## 3. Tiếp tục bằng hai slot

Trong PowerShell đã kích hoạt đúng môi trường ở bước 1:

```powershell
python -u .\ATKBackd\run_a1_three_tables_parallel.py `
  --outdir "$DOSE_OUT" --devices cuda:0 --jobs-per-gpu 2 `
  2>&1 | Tee-Object -FilePath "$DOSE_OUT.startup.log" -Append
```

Helper đọc `retrain_seed42`, đường dẫn dữ liệu và workers từ manifest cũ.
**Không truyền `--fresh`, không tạo output mới và không xóa checkpoint/cache.**
Hai slot là hai victim độc lập trên `cuda:0`, không phải DDP hoặc đổi batch size.

- Cell hoàn tất được audit rồi reuse/skip; không train lại chỉ để đổi lịch.
- Cell dở tiếp tục từ checkpoint hợp lệ: trọng số, optimizer, RNG và mốc epoch
  được khôi phục; lịch learning rate dùng cùng recipe gốc.
- Phần epoch đang chạy nhưng chưa lưu lúc dừng có thể phải chạy lại. Không hứa
  tiếp tục đúng minibatch hoặc RNG trajectory y hệt trên Windows multi-worker.
- Cell chưa có checkpoint bắt đầu mới theo đúng seed/config cũ. Checkpoint
  sai plan, input audit hoặc source hash phải báo lỗi, không bỏ qua kiểm tra.

Giữ cửa sổ train mở và không để máy sleep. Theo dõi GPU và epoch time trong
cửa sổ khác; đánh giá lợi ích bằng thời gian hoàn thành tổng số victim, không
chỉ VRAM hoặc thời gian một epoch của một process.

```powershell
python .\ATKBackd\run_a1_three_tables_parallel.py --outdir "$DOSE_OUT" --status

Get-Content "$DOSE_OUT.startup.log" -Tail 20

Get-ChildItem "$DOSE_OUT\seed*\*\console.log" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 2 |
  ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 10 }

nvidia-smi
```

Status là đếm artifact, không thay thế audit cuối. Chỉ gọi ba bảng hoàn chỉnh
sau khi đủ 39 cells hợp lệ và aggregate hoàn tất; không diễn giải dry-run hay
test CPU thành kết quả MM-Fi/GPU.

## 4. Nếu hai slot không ổn định

Nếu CUDA OOM, lỗi input/source/checkpoint, lock hoặc child exit: lưu traceback
và log của đúng cell, **không thử lại mù quáng**. Trước tiên xác minh controller
và toàn bộ descendants của lần chạy hai slot đã thoát bằng cùng quy trình
chụp/kiểm tra cây process. Không thay batch size hay giảm epochs để né lỗi.

Sau khi xử lý nguyên nhân hoặc xác nhận chỉ thiếu tài nguyên cho hai slot,
có thể quay lại **runner tuần tự gốc**, vẫn đúng output và recipe:

```powershell
python -u .\ATKBackd\run_a1_three_tables.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$DOSE_OUT" `
  --devices cuda:0 --num-workers 4 --retrain-seed42 `
  2>&1 | Tee-Object -FilePath "$DOSE_OUT.startup.log" -Append
```

Lệnh quay lại trên chỉ dành cho manifest Windows hiện tại có workers=4 và
`retrain_seed42=true`. Không chạy song song với helper. Resume chỉ giữ công việc
đã lưu; thời gian tăng tốc hai slot và khả năng tái lập số học GPU chưa được
đo trên máy Vân.

## Kiểm thử local trước khi chuyển tới máy Vân

Các suite chọn lọc đạt **182 tests**, bỏ qua một test cần CUDA thật: 39 tests
cho helper song song, 109 tests cho bộ ba bảng gốc và 34 tests checkpoint/RNG.
Test CPU với dữ liệu nhỏ đã lưu checkpoint sau epoch 0, dừng có kiểm soát,
resume đúng worker gốc tới epoch 49 và so trọng số khớp với lượt 50 epochs
không gián đoạn. Các test cũng kiểm tra khóa controller/worker, source hash,
plan hash, checkpoint không bị xóa, cache được audit trước khi bỏ qua và
cleanup chỉ tác động worker do launcher tạo.

Đây không phải kết quả benchmark MM-Fi hay xác nhận hai job nhanh hơn trên
RTX 4090. Test CPU không bảo đảm khớp bit với GPU/Windows multi-worker. Các
scientific source cũ được giữ nguyên; chỉ thêm helper, tests và hướng dẫn này.
