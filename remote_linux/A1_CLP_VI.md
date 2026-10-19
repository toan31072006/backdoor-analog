# Bổ sung CLP vào T5, không chạy lại bốn defense

## Material Passport

- Ngày: 2026-10-20, Asia/Bangkok.
- Yêu cầu: thêm một defense ngoài Fine-Pruning; chỉ adapted, không cải tiến.
- Phương pháp: CLP, *Data-Free Backdoor Removal Based on Channel Lipschitzness*,
  ECCV 2022. [Bài gốc](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6305_ECCV_2022_paper.php),
  [code tác giả](https://github.com/rkteddy/channel-Lipschitzness-based-pruning).
- Trạng thái: code/test local không phải kết quả full MM-Fi/GPU. Không lựa
  chọn defense vì biết điểm CLP trên A1; hiện chưa có điểm đó.
- Kiểm thử: **98 test local đạt** (exit code 0), gồm cơ chế, đúng HPELi,
  checkpoint/resume, full-identity synthetic NPZ, paired report và helper
  Windows parse/mock. Năm source SHA256 khớp bytes tải lại từ các commit
  đã pin; source scientific v1/v2 giữ nguyên. Không có task thật được tạo
  hoặc full MM-Fi/GPU run trong kết quả kiểm thử này.

## Thêm đúng một phương pháp, giữ nguyên các nguồn cũ

Bộ T5 mở rộng: **No defense + FT, ANP, NAD, FT-SAM, CLP**, ba seed 42/43/44.
Tổng **18 evaluation records**: 15 record của v2 + **3 CLP mới**. Các victim
A1/rho=.1 đã train 50 epochs trong T1–T3 được giữ nguyên; không train lại A1,
không sửa trigger hoặc chạy lại bốn defense để bổ sung CLP.

Sidecar `ATKBackd/run_a1_clp.py` và manifest `a1_clp.resolved.json` ghi vào
`runs/mmfi_a1_clp_s42_43_44_v1`. V1/v2 chỉ là nguồn đọc; source code đã khóa,
manifest, checkpoint và kết quả của chúng không bị sửa. Bảng FP pilot vẫn
được giữ, việc không chạy thêm FP không cho phép xóa hay giấu kết quả cũ.

## CLP khác Fine-Pruning và áp dụng thế nào cho HPELi

Fine-Pruning xếp hạng activation trên clean data rồi prune/fine-tune. Nhánh
CLP này chỉ dùng trọng số và BN running statistics để tính score theo source;
**không dùng clean bank/labels, không train, không fine-tune hay chọn ngưỡng
theo TEST**. Data-free nói về thao tác pruning, không phải đánh giá accuracy.

- Khóa **u=3** theo source defaults cho cả ba seed. Không chạy sweep 0..10
  của BackdoorBench để chọn mức đẹp nhất; không fallback dựa trên attack score.
- Score dùng singular value lớn nhất của mỗi output kernel nhân với
  `abs(BN.gamma)/sqrt(BN.running_var)`. Giữ công thức source không thêm epsilon;
  variance không dương hoặc giá trị không hữu hạn phải báo lỗi.
- Prune khi score **lớn hơn** mean + u × sample SD; zero cả gamma và beta,
  giữ convolution weights/running buffers nguyên vẹn. Nếu không có channel
  nào vượt ngưỡng, ghi nhận pruned=0, không tự tăng strength.
- HPELi có **10 cặp Conv2d→BN2d trực tiếp, 960 channels**. Hai BN2d sau khối
  SK/attention và hai BN1d được giữ nguyên và ghi rõ trong metadata. Không
  ghép giả BN sau attention với convolution cuối của một nhánh.
- Đây là **CLP thích nghi HPE, giới hạn coverage ở các cặp trực tiếp**, không
  phải full-architecture/native CIFAR reproduction. Không suy từ kết quả này
  ra CLP nói chung thất bại hay model kháng mọi CLP.

Nguồn/commit/SHA256 được khóa tại `defend/a1_clp_sources.json`; implementation
độc lập, không vendor source tác giả hoặc suy diễn license từ BackdoorBench.

## Chạy trên PowerShell máy Vân

**Đợi job v2 kết thúc trước** nếu muốn giữ tổng mức sử dụng đúng 2 job/GPU.
Helper CLP không tự dừng v2, T1–T3 hay YOLO. Chưa có dữ liệu/checkpoint trên
máy Admin không có nghĩa có thể launch từ đây thay máy Vân.

Sau khi code đã push:

```powershell
cd "$env:USERPROFILE\code"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'git pull failed' }

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-A1CLPTask.ps1 -JobsPerGpu 2
```

Helper tự dry-run read-only trước khi đăng ký task `MMFI-A1-CLP`. Mặc định đủ
42/43/44, một GPU, 2 job, workers=0. Chạy nền S4U như helper v2; đóng VS Code/SSH
không dừng task, nhưng sleep/shutdown/OOM/process bị kill vẫn làm gián đoạn.
Không auto-retry hoặc auto-start sau reboot; không mặc định hết hạn sau 72 giờ.

Muốn chỉ kiểm tra metadata, thêm `-DryRun`. Resume bằng cùng lệnh sau khi job
đã dừng; nếu task CLP cũ còn chạy, `-StopExisting` chỉ dừng CLP cùng user và
đúng output, không xóa checkpoint/lock. Full TEST pass bị ngắt có thể phải
evaluate lại, không đồng nghĩa train lại victim hoặc bốn defense.

Reference mặc định là
`runs/mmfi_a1_defenses_ft_anp_nad_sam_s42_43_44_v2`. Nếu chưa có reference,
CLP có thể được đánh giá riêng nhưng bảng tổng vẫn **PARTIAL**. Dùng đúng
reference path khi export; đường dẫn reference được khóa ngay lần đầu tạo
manifest CLP. Nếu đã khởi tạo không reference thì không thể âm thầm gắn thêm
reference vào cùng output; dùng output mới đã duyệt. Vì vậy cách mặc định
khuyến nghị là **chạy xong v2 rồi mới khởi tạo CLP**. Không copy/sửa record để
ép đủ bảng. Nếu bốn
defense chưa chạy, dùng helper v2 trước, không có trong batch CLP này.

## Xem log

```powershell
$DOSE_CLP = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_clp_s42_43_44_v1"
Get-ScheduledTask -TaskName MMFI-A1-CLP | Select-Object TaskName,State
Get-ChildItem "$DOSE_CLP.task.*.stdout.log" | Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Encoding UTF8 -Tail 20 }
Get-ChildItem "$DOSE_CLP.task.*.stderr.log" | Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Encoding UTF8 -Tail 20 }
Get-Content "$DOSE_CLP\seed42\a1\clp\console.log" -Encoding UTF8 -Tail 20 -Wait
```

Running state không chứng minh tiến trình đang tăng; xem log và thời gian
sửa file. Ctrl+C chỉ dừng theo dõi log. Không mở controller/exporter trùng
root đang có owner giữ lock.

Sau khi controller đã kết thúc:

```powershell
Get-Content "$DOSE_CLP\table5_defenses_with_clp.md" -Encoding UTF8
```

Nếu v2 còn thiếu lúc bảng được xuất, sau khi bổ sung xong v2 có thể audit và
xuất lại **vào sidecar** bằng lệnh dưới (không train/prune/evaluate model):

```powershell
$DOSE_SOURCE = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1"
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_a1_clp.py --source-outdir "$DOSE_SOURCE" `
  --outdir "$DOSE_CLP" --export-only
if ($LASTEXITCODE -ne 0) { throw 'CLP export/audit failed' }
```

Lệnh dùng reference đã khóa trong CLP manifest; không thay bằng một run khác.

## Metric và tính đủ bảng

Giữ bốn metric chính của v2: T-MPJPE ↑, Target-PCK ↓, Clean MPJPE ↓,
Triggered-GT MPJPE ↓ theo hướng defender. Clean toàn bộ 17 khớp; hai metric
target và Triggered-GT chỉ [2,3], không alignment, dose 1. Target-PCK là
joint-level %, threshold .2 × clean GT scale (5,12), không phải pose-level ASR.

CSV/JSON giữ đủ 10 metric báo cáo của v2, clean reference cùng hai khớp,
recovery gap, no-trigger controls, Clean PA-MPJPE và sáu-dose/DCE diagnostic.
Paired repair-minus-No-defense phải cùng seed, checkpoint victim, trigger và
input identities. Không ghép một A1 mới hoặc split khác vào delta.

Mean ± sample SD tính trên ba seed, không phải trên frame. Chỉ đủ 18 record
đã audit mới gọi bảng tổng hoàn chỉnh; thiếu reference thì không bịa paired
delta/SD. Không gọi target matching giảm mà pose thật vẫn hỏng là defense
thành công. Native compute/data budgets khác nhau phải được khai báo.
