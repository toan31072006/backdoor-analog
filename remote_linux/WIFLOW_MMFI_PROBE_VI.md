# WiFlow MM-Fi 3D: kiểm tra theo source gốc, chưa phải full experiment

**Scope mới:** người dùng hiện yêu cầu training thật và commit/push sau test.
Runner/recipe/launcher mới nằm trong [A1_WIFLOW_VI.md](A1_WIFLOW_VI.md).
`probe_wiflow_mmfi.py` vẫn chỉ là synthetic compute probe; không dùng lệnh
probe bên dưới thay cho `Start-A1WiFlowTask.ps1`. Các mô tả local-only/no-push
ở cuối tài liệu là trạng thái lịch sử trước yêu cầu mới, không phải quyền hiện tại.

## Cập nhật vận hành: tạm dừng Graph để thử WiFlow

Người dùng đã yêu cầu pause Graph riêng. Chưa kết nối được SSH tới Vân nên
assistant chưa suspend/kill hoặc launch process nào. Không kill worker Graph
khi controller còn chạy: controller có thể dừng luôn MetaFi++ khi worker
Graph thoát lỗi. Suspend theo đúng PID cho phép Resume sau; nó **không giải
phóng RAM/VRAM hoặc lock**, không phải checkpoint-based shutdown.

Xem PID/command hiện tại bằng PowerShell trên Vân, chỉ chọn Python worker có
`--_cell graphposefi__s42__clean` hoặc `--_cell graphposefi__s42__a1`, và đúng
manifest `mmfi_cross_backbones_s42_43_44_v1/cross_backbones.resolved.json`.
Trong Resource Monitor (`resmon.exe`), tab CPU, đối chiếu **PID** rồi chọn
Suspend Process. Sau này Resume Process chính PID đó. Không Suspend theo
tên chung `python.exe`, không chạm controller/MetaFi++, không xóa file lock.

Nếu dùng terminal-only, [Microsoft PsSuspend](https://learn.microsoft.com/en-us/sysinternals/downloads/pssuspend)
hỗ trợ `pssuspend <PID>` và `pssuspend -r <PID>`; không dùng process name vì
công cụ sẽ áp dụng lên tất cả process cùng tên. Kiểm PID/command lại ngay
trước thao tác; chưa tự tải hoặc chạy PsTools trong workspace này.

Sau khi code mới đã được chép sang repo Vân và kiểm VRAM còn trống, thử nhỏ:

```powershell
cd "$env:USERPROFILE\code"
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\probe_wiflow_mmfi.py --models wiflow --device cuda:0 `
  --batch-size 8 --warmup 2 --steps 5
```

Đây chỉ là CUDA smoke test. Nếu MetaFi++ đang train, thời gian đo chịu tranh
chấp tài nguyên; không lấy nó làm so sánh tốc độ công bằng hoặc ETA full run.
Chưa train WiFlow trên dữ liệu thật, chưa chạy A1 và không có model checkpoint
mới. Phần giới hạn source/loss/skeleton trước full training bên dưới vẫn áp dụng.

## Material Passport

- Ngày: 2026-10-22, Asia/Bangkok.
- Origin Skill: academic-research-suite / experiment-agent.
- Phạm vi người dùng yêu cầu: thử WiFlow dựa trên code tác giả, so với HPE-Li
  và MetaFi++; không tự dừng/thay GraphPose-Fi, không tự launch GPU trên Vân.
- Verification Status: source AST/hash đã đối chiếu online; model/loss và
  gradient/serialization đã test synthetic CPU. Chưa có full MM-Fi/CUDA,
  clean utility, attack hoặc defense result.
- Đây là technical probe, không phải một hàng đã hoàn thành trong T1--T3/T5.

## Nguồn và phần giữ nguyên

Nguồn: [WiFlow tác giả](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling),
commit `8ddb9a01772c368fc591f69e8414044b49d817e1`, file
`cross_dataset_test/WiFlow/wiflow.py` — không dùng bản default 15-joint 2D.
SHA256 file gốc:
`9233dfd54bfe58025ff5c9a5eeca7ccee27c39e6c57376d989c7ec53e0b1a49e`.

Repo có license Apache-2.0; giữ full license và attribution. Chỉ trích model/loss
definitions với import torch/math tối thiểu. Bỏ imports plotting/dataset và
side effects ngoài model; không import/execute toàn bộ script nguồn. Không
download/load `best_pose_model.pth` hoặc thực thi pickle của bên thứ ba.

- Kiến trúc, initialization, dropout, grouping18, attention width→height,
  **lấy time slice cuối** trước decoder và native loss giữ nguyên.
- Input `[B,3,114,10]`, output source `[B,17,3]` có tọa độ âm được.
- Wrapper duy nhất: output thành `(pose[B,1,17,3], features[B,1088])`, features
  là axial-attention map mean theo time rồi flatten; không dùng prediction
  giả làm latent feature. Không thêm/bớt parameter, không đổi state-dict keys.
- Giữ nguyên `models/factory.py`, `models/cross_backbone_factory.py`, trainer,
  attack, metric evaluator, scientific source map và các manifest cũ.
  WiFlow chưa được đăng ký vào full cross-backbone runner.

## Điểm cần khóa trước full GPU training

Author `SimplePoseLoss` mặc định Smooth L1 vị trí beta=.1 + .2 lần Smooth L1
bone-length beta=.05. Nó không phải ordinary MPJPE của batch hiện tại. Các arm
edges trong native bone loss neo tại joint9, trong khi A1 payload skeleton
neo tại joint8. Đây là khác biệt cần audit về skeleton/loss trước khi quyết
định protocol; chưa kết luận source sai, **không âm thầm sửa graph hoặc loss**.
Exception fallback về bone loss0 của tác giả cũng được giữ nhưng không được
dùng làm bằng chứng rằng mọi input/skeleton đều hợp lệ.

Full run cần adapter/recipe/selection rule mới, kiểm tra đơn vị/normalization,
trusted clean pilot và metrics/control đầy đủ. Nếu giữ author recipe hoặc dùng
common MPJPE phải công bố rõ adaptation; không lấy số trong README so trực
tiếp với các run HPE-Li/MetaFi++ của mình. Không lựa chọn theo attack TEST đẹp.

## Phép đo local đã chạy

CPU AMD64 Family25 Model80, Torch2.9.1+cpu, Python3.13.5, FP32, 2 threads,
batch2; một validation pass riêng, warmup2, đo5 forward/backward steps/model.
Common synthetic MPJPE, **không optimizer step, không raw dataset**; mỗi model
chạy lần lượt. Bảng là median thời gian forward/backward, không phải accuracy.

| Model | Parameters đếm local | Median s/step |
| --- | ---: | ---: |
| HPE-Li | 813427 | 0.12536 |
| MetaFi++ | 26517868 | 3.20969 |
| WiFlow MM-Fi 3D | 1060597 | 0.09068 |

Chỉ là một probe ngắn, có nhiễu CPU/order/cache. Không suy ra speedup GPU,
CUDA peak hoặc ETA full50epoch từ bảng này. Timing không gồm data I/O,
trigger generation, optimizer state/update, checkpoint hoặc full evaluation.
Các lần đo cùng seed không phải independent training seeds.

Artifact local ngoài source repo:
`E:/wifi/deliverables/wiflow_probe_20261022/source_audit.json` và
`compute_cpu_b2.json`. Không overwrite các kết quả cũ.

## Lệnh kiểm tra

Từ repo root, dùng đúng environment Torch/torchvision đã có:

```powershell
python -u .\ATKBackd\probe_wiflow_mmfi.py --audit-upstream
```

Lệnh trên chỉ tải **public pinned source text** để đối chiếu SHA/AST, không
execute module tải về, không đọc dữ liệu nghiên cứu hoặc chạm GPU.

CPU (mặc định không viết output):

```powershell
python -u .\ATKBackd\probe_wiflow_mmfi.py `
  --device cpu --batch-size 2 --warmup 2 --steps 5 --threads 2
```

**Chỉ khi người dùng chủ động cho phép đo riêng trên GPU Vân**, không có job
khác tranh GPU, và code đã được chuyển tới Vân:

```powershell
conda activate "$env:USERPROFILE\backdooranalog\envs\dose-backdoor"
cd "$env:USERPROFILE\code"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$WF_REPORT = "$env:USERPROFILE\backdooranalog\runs\wiflow_compute_probe\gpu_b32_$(Get-Date -Format yyyyMMdd_HHmmss).json"

python -u .\ATKBackd\probe_wiflow_mmfi.py `
  --device cuda:0 --batch-size 32 --warmup 3 --steps 10 `
  --output "$WF_REPORT"
```

Đây **một process, ba model chạy tuần tự**, để đo riêng compute, không phải
launcher hai training jobs. Không tự kill/stop A1/GraphPose-Fi/YOLO, đổi batch
sau OOM hay fallback CPU. CUDA peak báo allocated tensor memory của probe,
không bao gồm optimizer state, toàn bộ driver/context hoặc VRAM job train đầy
đủ. Report phải là path mới, không nằm trong historical experiment output.

## Test và giới hạn

26 tests WiFlow đạt, exit0: exact source/wrapper forward/gradient/state parity
ở train/eval, signed17x3 output, attention order/last-time decoder, native loss,
temporary hook cleanup, invalid-input rejection, optimizer/RNG CPU roundtrip,
report guards và AST audit rejection. Online source audit cũng đạt với URL/SHA
pinned thật, không chỉ synthetic fixture. Chưa kiểm CUDA, full-data convergence
hoặc full-run resume; historical scientific sources không đổi.

Lượt gộp với cross-backbone runner/adapter/evaluator cũ đạt **103 tests, exit0**
(26 WiFlow +77 regression). Source/recipe/evaluator cũ vẫn không bị sửa.
Theo phản hồi người dùng, **chỉ kiểm thử local, không commit/push** bản này.
