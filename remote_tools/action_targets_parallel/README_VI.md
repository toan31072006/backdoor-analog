# Cross/Nod: MM-Fi + Person-in-WiFi-3D, 2 job trên 1 GPU máy Vân

## Material Passport

- academic-research-suite / experiment-agent; implementation handoff, 2026-11-08.
- Người dùng duyệt push GitHub, cả hai dataset, tối đa2job/cuda:0 và evaluate
  ngay khi từng victim train xong. Không tự launch từ máy người hỗ trợ.
- Bốn victim HPE-Li scratch, seed42/rho.1; target Cross hai tay bắt chéo trước
  ngực, Nod cúi đầu quanh cổ. Không phải chỉ thay trigger motion prior.
- Separate exploratory target-action extension, không phải full T7 có baseline
  và Clean controls. Không sửa paper hoặc lấy hình synthetic làm kết quả thật.
- Xem VALIDATION.md cho kiểm thử local; chưa có full-data/CUDA run mới.

## Cross v2: sửa target quá xa trên pose có tay ngắn (2026-11-08)

Người dùng duyệt đổi Cross sang target thích nghi tầm với. Nod và toàn bộ
recipe/poison/trigger/metric/lịch2slot giữ nguyên. Git hiện dùng Cross geometry
v2 cho CẢ MM-Fi/PiW; hai ZIP ngày07/11 và output v1 giữ nguyên, không sửa/chép
plan hoặc checkpoint cũ. Chi tiết quy tắc và lỗi TRAIN9092:
[CROSS_REACH_V2_VI.md](CROSS_REACH_V2_VI.md).

**Nếu đã chạy Geometry v1 bị lỗi:** sau pull, đổi `$CrossNodArgs.Outdir` sang
`U:\backdooranalog\runs\cross_nod_mmfi_piw_s42_2jobs_v2`, rồi tự chạy Geometry
lại. Không Resume/Prepare output v1, không xóa output/lock. Geometry mới vẫn
kiểm mọi poison TRAIN pose và báo lỗi nếu thật sự không thể bắt chéo. Xem
`records.cross.reach_adaptation` trong geometry.json: số pose/arm phải điều
chỉnh, độ điều chỉnh lớn nhất, khoảng sang phía đối diện/phía trước nhỏ nhất.
Chưa có bảo đảm full dữ liệu thật đạt chỉ từ local tests.

## 1. Pull code và kiểm đường dẫn (chưa train)

Trong terminal SSH của VS Code trên Vân, dùng checkout đã pull:

```powershell
Set-Location -LiteralPath 'C:\Users\Van\PiWMetaFiTwoExtra4090'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; preserve local changes, do not reset.' }
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$CrossNodRunner = '.\remote_tools\action_targets_parallel\Run-CrossNodTwoJobs.ps1'
$CrossNodArgs = @{
    DataHome = 'C:\Users\Van\backdooranalog'
    Python = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
    Outdir = 'U:\backdooranalog\runs\cross_nod_mmfi_piw_s42_2jobs_v2'
}
& $CrossNodRunner -Mode CheckOnly @CrossNodArgs
```

Không cần ZIP/pip install, không đè source cũ trong DataHome. Nếu không có U:
chọn disk local khác còn ít nhất8GiB, giữ basename output như trên. Không dùng
thư mục output serial v1 hoặc copy checkpoint cũ vào đây. Không git pull trong
lúc run mới đang chạy: source/runtime đã được hash vào plan/checkpoint.

Mặc định cần:

```text
C:\Users\Van\backdooranalog\datasets\Compress
C:\Users\Van\backdooranalog\datasets\Person-in-WiFi-3D_processed_v1
C:\Users\Van\backdooranalog\actions\data_bend.npy
C:\Users\Van\backdooranalog\actions\data_cross.npy
C:\Users\Van\backdooranalog\actions\data_nod.npy
```

Nếu PiW nằm chỗ khác thêm PiWDatasetRoot vào hashtable; nếu prior nằm chỗ khác,
thêm CrossAction/NodAction. Dùng nguyên hashtable cho mọi mode. CheckOnly dùng
stdlib, không Torch/CUDA, không tạo output hoặc download dataset.

## 2. Geometry TRAIN-only: xem trước khi duyệt

Nếu đã xác nhận hai NPY thật đúng Cross/Nod nhưng thiếu provenance sidecar:

```powershell
$CrossNodArgs.ApproveActionLabels = $true
& $CrossNodRunner -Mode Geometry @CrossNodArgs
```

Không dùng cờ trên để đoán nhãn file hoặc copy Bend đổi tên. Với sidecar hợp
lệ có thể bỏ cờ; quyết định phải giữ cố định từ Geometry đến hết run.

Xem bốn ảnh trong hai thư mục:

```text
OUT\runs\mmfi_action_targets_cross_nod_s42_v2\geometry_cross.png
OUT\runs\mmfi_action_targets_cross_nod_s42_v2\geometry_nod.png
OUT\runs\piw_action_targets_cross_nod_s42_v2\geometry_cross.png
OUT\runs\piw_action_targets_cross_nod_s42_v2\geometry_nod.png
```

Đây là GT/target của TRAIN, không có model predictions. Audit mọi poison pose
qua sáu dose, không loại pose lỗi hoặc tune geometry trên TEST. Phải xác nhận
hai tay thực sự phía trước ngực, đầu cúi đúng hướng. Nếu sai: dừng trước Prepare,
gửi ảnh/lỗi; đổi geometry phải có version/output mới, không sửa plan đã train.

## 3. Prepare: GPU rảnh, chỉ audit và smoke, chưa full train

Sau khi đã xem đúng geometry của CẢ HAI dataset:

```powershell
& $CrossNodRunner -Mode Prepare -ApproveGeometry @CrossNodArgs
```

Full-content input/identity audit; single real CUDA B32 smoke/action; rồi hai
process B32 đồng thời MM-Fi Cross/Nod và hai process B32 đồng thời PiW Cross/Nod.
Barrier giữ cả hai model/optimizer resident khi smoke. Tất cả weights bị bỏ,
không dùng warm-start. Chỉ ghi receipt chung sau cả hai concurrent pairs đạt.
Prepare không tự chạy training. Smoke chứng minh capacity lúc đó, không bảo đảm
không OOM về sau hoặc tốc độ gấp2. Không tự đổi batch/AMP, retry hay kill job.
Chờ job GPU khác xong; guard chỉ cho phép peer do chính coordinator vừa tạo.

## 4. Launch: 2 slot, evaluate ngay trong từng job

Chỉ khi Prepare báo PARALLEL_PREPARED_ONLY:

```powershell
& $CrossNodRunner -Mode Launch @CrossNodArgs
```

Thứ tự cố định:

1. MM-Fi Cross và MM-Fi Nod song song trên cuda:0.
2. Job nào đạt final49 trước sẽ evaluate sáu dose ngay, dù peer còn train.
3. Sau cả hai MM-Fi evaluate hoàn tất mới mở PiW Cross và PiW Nod song song.
4. Mỗi PiW job evaluate ngay sau final199, không chờ peer train xong.

**Evaluate vẫn chiếm slot**: không có trường hợp2job train +1job eval. Không
thay model bằng MetaFi++; đây là các gói target-action HPE-Li đã bàn trước.
MM-Fi: P1S1/split0, TRAIN133056/TEST33264, SGD.001/momentum.9/wd0,
50epoch/final49, B32/FP32/workers4. PiW: TRAIN27876/TEST2586, AdamW.001/wd.01,
200epoch/final199, B32/FP32/workers0/drop_lastTrue. Mỗi process RNG/BN/optimizer
riêng, scratch; không warm-start từ leg target/Clean/smoke/peer.

Giữ terminal SSH và máy thức; chưa cấu hình Windows Scheduled Task/detached
service. Log riêng từng job trong OUT\sessions\SESSION\*.job.log; terminal
coordinator in PID, slot, heartbeat và bảng partial sau mỗi job complete.

## 5. Xem tiến độ, Resume hoặc Evaluate-only

```powershell
& $CrossNodRunner -Mode Status @CrossNodArgs
```

Status chỉ là phase/PID/log đã lưu, không xác nhận PID đang sống. Mỗi epoch có
atomic checkpoint đủ model/optimizer/RNG/config/input/source/runtime binding.
Nếu lỗi: coordinator không mở job mới, đợi peer nó sở hữu hoàn tất, không kill.
Giữ log/checkpoint, xem lỗi rồi mới tự quyết định:

```powershell
& $CrossNodRunner -Mode Resume @CrossNodArgs
```

Resume chỉ checkpoint đúng plan; job complete được verify và không train lại.
Nếu chỉ muốn evaluate các fixed-final checkpoint đã có (KHÔNG train):

```powershell
& $CrossNodRunner -Mode Evaluate @CrossNodArgs
```

Evaluate-only vẫn kiểm từng final checkpoint, báo lỗi nếu chưa đủ epoch. Không
đổi intermediate thành completed. Trường hợp orphan NPZ/partial geometry/capacity
failure: không tự xóa artifact/lock, không auto-retry; gửi log để kiểm tra.
Ctrl+C ngừng mở job mới nhưng đợi các job đã sở hữu xong để giữ lock an toàn.
Nếu coordinator bị force-kill, worker phát hiện lease mất ở ranh giới epoch
hoặc trước evaluation; slot OS lock chặn một controller mới chạy trùng.

## 6. Kết quả và qualitative thật

Sau MỖI job complete đã có ACTION\results.json, complete.json và
poses_full_test.npz. OUT\partial_results.csv/.json cập nhật ngay, không gọi bảng
thiếu hàng là complete. Sau4/4 mới có OUT\results.csv/.json; không lấy mean của
hai dataset hoặc của Cross/Nod có joint sets và scales khác nhau.

MM-Fi targets Cross[12,13,15,16], Nod[10], PCK scale cleanGT(5,12).
PiW targets Cross[6,10,4,8], Nod[1], PCK scale cleanGT elbows(6,4).
T-MPJPE/PCK tại d1, DCE mean5dose dương; full six-dose requested/predicted/DCE,
same-victim no-trigger target controls, clean all-joint MPJPE/PA và target-joint
clean error. Giữ ordered IDs và raw GT/target/prediction XYZ để vẽ qualitative.
PiW có PNG/PDF qualitative tự xuất với fixed TEST indices; MM-Fi giữ full NPZ
và fixed TEST indices cho plot tiếp. Không dùng AI để giả ảnh prediction.

Cross là bilateral two-bone IK + bone-direction interpolation, không phải
rigid subtree40degree; không áp định lý monotonic/near-linear của payload cũ.
Bảo toàn bone lengths không chứng minh joint limits/collision/contact. Nod
movement nhỏ có thể PCK cao dù không phản ứng: luôn đọc cùng no-trigger/DCE/Λ.
Chưa có clean-trained+trigger controls cho các target mới, không suy từ scalar
baseline cũ. Một seed không có seed SD, không phải full T7 hay physical attack.
