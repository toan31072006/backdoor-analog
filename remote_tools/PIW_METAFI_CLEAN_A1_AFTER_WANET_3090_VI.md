# PiW3D MetaFi++: nối Clean/A1 sau WaNet trên 3090

## Phạm vi

Pilot seed42, Person-in-WiFi-3D one-person, không phải MM-Fi hay full 8-row
comparison. Chỉ thêm Clean rồi A1, tối đa một GPU worker. Không thêm TSBA,
Smooth LF hay train lại bốn baseline. Không dừng hoặc sửa checkout đang train.
`Queue` cho phép đợi queue cũ kết thúc, tự Prepare/CUDA smoke, train hai victim
và evaluate. Đây là quyền chạy queue được chọn trước, không phải Prepare-only.

- Giữ nguyên frozen source bundle/entry point của run hiện hành: MetaFi++
  ResNet34 ImageNet initialization then metafi_init; AdamW lr0.01/wd0.01,
  physical/effective B32, accumulation1, workers0, FP32, 200epoch/final199.
- Giữ official TRAIN27876/TEST2586, preprocessing, action bend SHA gốc,
  rho0.1/paired doses, target `piw_body_hip_axis_v2` pivot7->[11,13], axis
  normalize(clean_GT[5]-clean_GT[7]), 40deg*d, PCK ref(6,4), threshold0.2.
  Clean không poison; A1 giữ native operator. Không dùng MM-Fi/Z-axis hay
  checkpoint MetaFi SGD thay thế PiW AdamW.
- Native plans của baseline và Clean/A1 có profile/method list/mô tả khác nhau
  theo thiết kế gốc; sidecar chỉ cho phép các metadata/path đó khác. Toàn bộ
  recipe, target, trigger, poison, ordered input/content và shared source phải
  khớp. Không sửa hash/manifest cũ để vượt gate.
- Output mới `runs/piw_metafi_clean_a1_after_wanet_s42_3090_v1`, checkpoints:
  `clean_a1/attacks/metafiplusplus/seed42/{clean,a1}/checkpoint.pt`.
  Matching AdamW có thể resume sau native source/config/poison/optimizer/RNG
  audit; SGD bị loại. Chưa có phù hợp thì victim mới theo recipe đã khóa.

## Dùng checkout riêng khi queue cũ còn chạy

Không pull trong `piw4`, snapshot/task hoặc source đang chạy. Máy 3090 trước
đây không có git trong PATH; có thể lấy GitHub ZIP source nhỏ (không dataset/
checkpoint) từ `https://github.com/toan31072006/backdoor-analog/archive/refs/heads/main.zip`,
giải nén vào thư mục mới `C:\Users\ssh3090\piw_after_wanet\backdoor-analog-main`.
Không extract đè source hiện có. Giữ nguyên môi trường `eclare-env` đã chạy GPU.

Trong SSH PowerShell thứ hai, sau khi lấy source:

```powershell
$S = 'C:\Users\ssh3090\piw_after_wanet\backdoor-analog-main\remote_tools\Start-PiWMetaFiCleanA1AfterWaNet3090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Queue
```

Giữ cả cửa sổ SSH queue cũ và cửa sổ chờ mới mở. Chờ không dùng CUDA; in PID
queue cũ và danh sách chưa có final199 mỗi30s. Chỉ khi old queue không còn
process và đủ bốn receipt train final199 mới lấy native output lock, kiểm
SHA256 checkpoint + full AdamW/model/RNG/header/plan, rồi Prepare/training.
Nếu queue cũ crash trước khi xong WaNet thì dừng, không mở Clean/A1 sớm.
Không lấy file tồn tại hoặc GPU-util thấp làm bằng chứng completion.

Queue nối giữ lock queue cũ suốt lượt mới để tránh mở lại baseline chồng GPU.
Mọi child chạy blocking tuần tự; check GPU idle trước từng phase. Lỗi dừng
toàn bộ pending, không retry/OOM fallback/shrink batch/kill process. Ctrl+C
trong queue nối sẽ chờ current owned worker thoát, không gửi kill; giữ SSH
trong lúc chờ. Mất SSH hoặc reboot không được bảo đảm sống tiếp; không tmux,
Task Scheduler, WMI, service hoặc credential/policy changes.

## Theo dõi và kết quả

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Status chỉ đọc marker/log, không load Torch/CUDA hay train. Xem log dưới
output mới `logs/{prepare,check,train,evaluate}.{clean,a1}.*.log`.
Checkpoint final199, predictions.npz, evaluation.json riêng từng cell và
`clean_a1_results.json` được lưu; six-dose, target metrics/clean utility và
same-seed Clean controls dùng native evaluator. Clean attack columns không
xếp hạng. Không evaluate/sửa bốn baseline tự động trong task này.

Một seed chưa có mean/std nhiều seed; đây là bổ sung selected-method, không
gọi full comparison. Local CPU/mock tests không thay cho GPU smoke/dataset
audit hoặc efficacy trên 3090. SHA source và gate thực được bind khi enqueue;
không cập nhật checkout mới khi waiter/worker đang chạy.
