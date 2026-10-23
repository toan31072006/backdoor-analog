# WiPose: đo overhead trước khi thay runner

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: bounded implementation diagnostics, không phải thực nghiệm paper.
- Origin Date: 2026-10-24, Asia/Bangkok.
- Verification Status: local CPU/synthetic checks only until a separate CUDA report exists.
- Version Label: wipose_speed_diagnostic_v1_not_a_paper_experiment.
- Người dùng chọn **giữ nguyên hai job Clean/A1 đang chạy trên Vân** trong lúc
  chuẩn bị benchmark. Không dừng task, không pull/ghi đè runner đang chạy,
  không xóa khóa hoặc checkpoint, không tự launch lại training.
- Gate mới: **60 tests passed**, exit0, 18.05s trên CPU/synthetic; có real
  HPE-Li B2/two-step bitwise parity, NaN/Inf fail-before-update, source-tamper
  refusal và CUDA visible-client refusal mocks. Một smoke CPU B32 cũng hoàn
  thành. Không có số đo CUDA/real-data throughput hoặc thay đổi production.

## Phạm vi

`ATKBackd/probe_wipose_speed.py` là sidecar độc lập, không nằm trong source
bindings của run WiPose đang chạy. Chỉ thêm script này không làm đổi hash của
runner/model/loader cũ. Không commit/push tự động hoặc chuyển production sau test.

Con số user cung cấp: Clean epoch0 có `t=1434.8s`, 130496 mẫu/B32 =4078 batch,
tức khoảng .352s/batch. Timer gồm đọc dữ liệu, train và lưu checkpoint cuối epoch,
không gồm preflight trước đó. MM-Fi có số batch tương đương và input lớn hơn;
không giải thích mức chậm chỉ bằng số mẫu.

Các ứng viên cần đo, chưa phải nguyên nhân đã định lượng:

1. WiPose dùng Python kiểm tra từng gradient CUDA mỗi batch; candidate gộp
   predicate trên device rồi mới đọc một boolean. Vẫn kiểm loss trước backward,
   và vẫn từ chối gradient NaN/Inf trước optimizer update. Không bỏ safety check.
2. Đọc/kiểm tra/min-max từng NPZ với workers0. Bản đo cache giữ nguyên kết quả
   FP32 của `read_frame`, không đổi precision, normalization hoặc poison input.
3. Hai job chia sẻ GPU và ổ đĩa. Không lấy benchmark đang tranh GPU để hứa ETA.

## Chạy khi hai job vẫn đang chạy

Chỉ xem metadata; không truy cập dataset, model hay CUDA:

```powershell
$WIP_PY = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
& $WIP_PY -u .\ATKBackd\probe_wipose_speed.py --mode plan
```

Nếu cần đo I/O ngay, người dùng có thể chủ động chạy một lượt nhỏ TRAIN-only.
Nó không dùng GPU nhưng vẫn tranh CPU/ổ đĩa với training. Kết quả lúc đó là
**contention measurement**, không phải cold-disk benchmark hay full-epoch speedup:

```powershell
& $WIP_PY -u .\ATKBackd\probe_wipose_speed.py --mode io `
  --dataset-root "$env:USERPROFILE\backdooranalog\datasets\WiPose_processed_v1" `
  --samples 128
```

Probe chọn 128 identities không lặp bằng RNG riêng; đọc TRAIN, không đọc TEST.
Nó báo selection hash, coverage, payload bytes, thời gian đọc lần đầu và lookup
RAM, đồng thời re-read để kiểm bitwise. Không tạo cache trên ổ đĩa. Không suy
thời gian cả epoch từ một cache nhỏ; OS cache không được kiểm soát.

## Chỉ đo CUDA khi host đã rảnh

Không chạy lệnh này lúc hai job còn hoạt động. Sidecar sẽ từ chối khi
`nvidia-smi` thấy process compute khác; nếu không đọc được trạng thái, nó cũng
từ chối. Guard chỉ là snapshot, không phải reservation; không mở job khác trong
lúc đo. WDDM có thể không hiện đầy đủ compute clients, vì vậy người chạy vẫn
phải xác nhận các task training đã idle; không lấy output rỗng làm chứng minh
GPU chắc chắn rảnh. Không có flag bỏ qua guard hoặc tự kill process.

```powershell
& $WIP_PY -u .\ATKBackd\probe_wipose_speed.py --mode compute `
  --device cuda:0 --steps 20 --warmup 2 --rounds 2
```

Hai disposable HPE-Li model/synthetic batches, B32/FP32/SGD .01/momentum .9,
deterministic/no TF32 như runtime cũ. Trước khi đo, probe kiểm hai bước B2:
output, loss, gradient, parameters/BN buffers, optimizer và Torch RNG phải
bitwise bằng nhau. Mỗi timing case warm-up riêng; thứ tự legacy/grouped đảo
giữa hai round. Không đọc checkpoint hoặc train trên dataset thật. Có giới hạn
steps/rounds; timeout được kiểm giữa các step, không thể ngắt một CUDA op bị treo.

Trên máy chỉ có CPU có thể dùng `--device cpu --steps 2 --warmup 1 --rounds 1`
để kiểm wiring/parity. **CPU time không chứng minh tốc độ trên 4090**.

## Sau khi có kết quả

- Chỉ chọn tối ưu nếu có số đo CUDA/I/O và parity phù hợp; không hứa nhanh2x.
- Không thay batch, loss, model, LR, epoch, split, seed, poison/dose hoặc metric.
- Probe pass không chứng minh full-data numerical equivalence hoặc GPU VRAMfit.
- Run cũ khóa source/input/runtime trong checkpoint. Không sửa manifest/hash,
  không chép checkpoint sang profile mới để lách validation. Nếu cần runner mới,
  phải thiết kế provenance/resume hoặc run mới riêng và xin người dùng quyết định
  trước khi thay đổi production. Giữ nguyên mọi artifact lịch sử.
- Hai job hiện tại tiếp tục cho đến khi người dùng yêu cầu khác. Benchmark
  không biến pilot thành full comparison/defense hay kết quả nhiều seed.
