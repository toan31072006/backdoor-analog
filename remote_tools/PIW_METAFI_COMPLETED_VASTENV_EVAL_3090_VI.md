# Clean mới trên GPU và evaluate checkpoint cũ trên CPU

## Phạm vi đã được người dùng xác nhận, 2026-11-01

- Skill: academic-research-suite / experiment-agent; kiểm tra tái lập, không phải bảng paper mới.
- Người dùng đã gửi smoke Clean B32 PASS trong env Vast-version: loss 3.876457929611206 so với log Vast 3.8719141483306885, lệch tương đối đối xứng 0.11721477088165644%.
- Người dùng yêu cầu train đủ Clean rồi evaluate và chọn **CPU evaluate song song với Clean GPU**.
- Chỉ Clean mới được train. FTrojan, BPP, FIBA, WaNet, Clean cũ và A1 cũ chỉ đọc checkpoint final199; không train/resume hoặc chọn checkpoint theo TEST.
- Chưa có kết quả Clean mới hoặc kết quả CPU mới chỉ vì code đã được tạo/test. Smoke gần nhau không bảo đảm hội tụ giống nhau.
- Kiểm thử local: **217 passed** (evaluator mới và hồi quy evaluator FTrojan),
  CPU/toy/mock với source/hash thật; không phải chạy full TEST/checkpoint của
  máy 3090. Không có cài thư viện, train hoặc CUDA mới từ máy Admin.

## 1. Terminal SSH thứ nhất: train Clean đến final199 rồi tự evaluate

Giữ nguyên checkout đã Prepare, không pull/sửa runner đã được gate khóa hash.
Không cài lại env và không chạy lại smoke để chọn loss thuận lợi hơn.

```powershell
$Rev = 'a63b3fcda81aa6ffe786c23881bcf941b87113e4'
$Repo = "$env:USERPROFILE\code_clean_vast211_a63b3fc\backdoor-analog-$Rev"
$S = "$Repo\remote_tools\Start-PiWMetaFiCleanVastEnv3090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
if ($LASTEXITCODE -ne 0) { throw 'Clean launch loi; dung, khong tu retry.' }
```

Foreground: giữ SSH này mở. Một Clean worker trên cuda:0, AdamW LR0.01/WD0.01,
B32/FP32, seed42, 200 epoch/final199 như recipe đã khóa. Lưu checkpoint atomic
mỗi epoch cùng optimizer/RNG. Chỉ resume chính run này sau khi mọi binding khớp.
Launch kiểm lại gate đã Prepare; không sửa gate, giảm batch hoặc fallback.

Kết quả Clean mới nằm riêng ở:

`backdooranalog\runs\piw_metafi_clean_vastenv_s42_3090_v1\comparison.json`

Log tiến độ nằm ở `logs\launch.*.log`; trong SSH khác có thể dùng cùng launcher
với `-Mode Status`. GPU 0% không tự chứng minh đã train/evaluate xong.

## 2. Terminal SSH thứ hai: evaluate checkpoint cũ bằng env mới, chỉ CPU

Dùng script **Eval-PiWMetaFiCompletedVastEnv3090.py** tại commit được bàn giao
trong chat. Tải riêng script này về đường dẫn mới; không cập nhật checkout đã
Prepare hoặc bất kỳ source/gate nào đang dùng cho Clean.

Sau khi tải script về `$E`, chạy từng method bằng một process riêng:

```powershell
$ErrorActionPreference = 'Stop'
$Rev = 'a63b3fcda81aa6ffe786c23881bcf941b87113e4'
$Repo = "$env:USERPROFILE\code_clean_vast211_a63b3fc\backdoor-analog-$Rev"
$H = "$env:USERPROFILE\backdooranalog"
$Python = "$H\envs\piw-metafi-vast211\Scripts\python.exe"

foreach ($Method in 'ftrojan','bpp_source','fiba','wanet_source','clean','a1') {
    & $Python -u $E --repo $Repo --data-home $H --method $Method --device cpu
    if ($LASTEXITCODE -ne 0) { throw "CPU eval $Method loi; dung, khong ghi de/retry." }
}
```

Không gộp nhiều profile native trong một interpreter: chúng dùng các global
mutable khác nhau. Vòng lặp trên chỉ có một process evaluate CPU mỗi lúc, không
thêm job GPU và không kill job đang chạy. CPU/RAM/disk vẫn chia sẻ với Clean,
vì vậy không hứa tốc độ train không đổi hoặc có kết quả ngay lập tức.

Nguồn checkpoint lịch sử mặc định:

| Method | Run gốc | Nhánh dưới run |
| --- | --- | --- |
| FTrojan/BPP/FIBA/WaNet | piw_metafi_four_attacks_s42_3090_v1 | baseline_six/attacks/metafiplusplus/seed42/METHOD |
| Clean cũ/A1 cũ | piw_metafi_clean_a1_after_wanet_s42_3090_v1 | clean_a1/attacks/metafiplusplus/seed42/METHOD |

Các checkpoint, plan, gate, receipt và prediction/evaluation lịch sử không được
ghi đè, migrate hoặc thay runtime để vượt kiểm tra. A1 phải khôi phục được đúng
trigger đã ghi trong plan; nếu khác key/array thì dừng và báo thiếu điều kiện,
không âm thầm tái tạo một trigger khác. Một method lỗi không biến các method
chưa evaluate thành COMPLETE.

Kết quả CPU được ghi riêng ở
`backdooranalog\runs\piw_metafi_completed_vastenv_eval_s42_3090_v1\cpu\METHOD`,
không nằm trong output train lịch sử. Chỉ reuse cache khi mọi binding khớp;
artifact dở/sai được giữ nguyên, không tự xóa. Lưu full 2586 TEST identities,
GT, predictions; attack có sáu dose và target, Clean cũ chỉ có một no-trigger
pass thật (dose0), không giả tạo six-dose Clean-trigger controls.
Mỗi result phân biệt **runtime của checkpoint/plan train lịch sử** với
**runtime evaluate hiện tại**; các continuation từ Vast còn có provenance cũ.

## 3. Diễn giải kết quả đúng

- Clean mới: train bằng env 2.11 rồi evaluate; đây là thử nghiệm hội tụ mới.
- Các victim cũ: giữ nguyên learned state, chỉ đổi runtime evaluate. Đây là
  kiểm tra inference, không biến chúng thành model đã train bằng env 2.11.
- Clean cũ được evaluate lại để phân biệt thay đổi inference với thay đổi
  training trajectory. Không thay Clean cũ bằng Clean mới trong các gate/control
  lịch sử hoặc giả rằng mọi victim được train cùng runtime.
- Năm metric victim không cần giả tạo prediction từ một Clean-trained model
  khác. Clean-triggered reference controls còn thiếu phải ghi rõ, không dùng
  cùng victim làm clean reference. Với dòng Clean, không xếp hạng attack columns.
- Đơn vị Clean/T/DCE là mm; Target-PCK là % trên target joints, không pose ASR.
  Mean DCE chỉ lấy năm dose dương. Không sửa metric, target, seed hay chọn best
  checkpoint/đổi hyperparameter theo TEST.
- Đây là một seed; không mean/std nhiều seed, không bằng chứng một thư viện
  cụ thể gây lỗi. Windows/GPU/driver và policy luồng vẫn khác Linux/Vast.
