# Cross/Nod target-action extension — MM-Fi / HPE-Li

**Cross reach v2 (08/11):** Git đã sửa goal quá xa theo tầm với từng pose;
Nod/recipe không đổi. Mặc định output mới v2; không dùng lại plan/output v1.
ZIP07/11 giữ nguyên. Xem [quy tắc và lệnh khôi phục](../action_targets_parallel/CROSS_REACH_V2_VI.md).

**Cập nhật 2026-11-08:** để chạy CẢ MM-Fi/PiW với2job/1GPU và evaluate ngay từng
job, dùng [coordinator mới](../action_targets_parallel/README_VI.md). File này
giữ hướng dẫn serial của gói ZIP cũ, không trộn hai launcher/output.

## Material Passport

- Origin: academic-research-suite / experiment-agent; code implementation requested by user.
- Scope: **hai target action mới**, không phải thay file trigger rồi vẫn xoay chân.
- Verification: code/tests only. Chưa có full-data/CUDA training hoặc kết quả mới.
- Data: MM-Fi P1S1/split0, TRAIN133056/TEST33264; seed42, rho.1 (13305 poison).
- Victim: scratch HPE-Li, SGD .001/momentum.9/wd0, physical32,50epochs/final49,
  ordinary all-joint MPJPE, FP32,4loaderworkers. Không pretrained/early-stop/best TEST.
- A1 trigger native md_multicarrier/eps.185, paired fixed U(.2,1), no cover.
  Mỗi action dùng prior tương ứng data_cross.npy hoặc data_nod.npy.
- Chạy serial Cross -> Nod; mỗi target là victim độc lập. Không train lại Clean,
  Bend, baseline hay defense. Không sửa source/run/manifest đang chạy trên Vân.
- Đây là **exploratory target-action extension**, không phải T7 đầy đủ có baseline
  và không được chép vào bảng motion-prior cũ. Một seed không có seed uncertainty.

## Hình học cần duyệt trước khi chạy

- **Nod:** neck9 làm pivot, xoay head10 theo lateral axis lấy từ hai vai,
  góc40*d độ. Chỉ khớp10 đổi; cổ/thân/tay/chân giữ nguyên trong target.
- **Cross:** vai11/14 giữ nguyên, hai khuỷu/cổ tay12/13/15/16 di chuyển. Endpoint
  hai cổ tay nằm hai phía đối diện của đường giữa thân và phía trước ngực.
  Dùng two-bone IK bảo toàn upper/lower arm lengths, elbow pole xuống/ra ngoài.
  Dose nội suy hướng hai xương bằng great-circle rotations; dose0 identity.
- Cross **không phải rigid single-subtree rotation** của công thức cũ; góc từng
  xương có thể khác nhau theo pose. Không claim monotone wrist displacement hoặc
  sai lệch tuyến tính2% của payload40deg cũ cho Cross. Geometry audit đếm cả
  các bước giảm displacement, không sửa/xóa chúng để làm đẹp curve.
- Cả hai giữ toàn bộ bone lengths nhưng không bảo đảm joint-angle limits,
  self-collision, contact consistency, hay được action classifier nhận đúng.
- Body forward là cross(lateral,up), convention công khai từ joint layout.
  **Xem PNG TRAIN trước:** nếu Cross không đúng phía trước hoặc Nod sai hướng,
  không Prepare/Launch; gửi hình để sửa geometry/version/output trước training.
- `pivot`, `payload_axis`, `theta_max_deg` trong resolved cfg là compatibility
  fields cho native recipe. **action_payload mới là định nghĩa target thật**;
  không đọc các field tương thích ấy để gọi Cross là một-arm40deg.
- Run cũ tên bend dùng leg rotation; không tự đổi tên thành target trunk bend
  hoặc ghép thành ba semantic actions đã được train. Script này chỉ thêm Cross/Nod.

## Dùng trên máy Vân

Giải nén code ZIP vào thư mục MỚI, ví dụ `C:\Users\Van\Downloads\MMFiCrossNod`.
Không giải nén đè checkout/job đang chạy. Không cần pip install/git pull và
không có checkpoint hoặc raw data trong ZIP. Dùng Python env đang có:
`C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.

Mở PowerShell trên Vân:

```powershell
Set-Location "$env:USERPROFILE\Downloads\MMFiCrossNod"
$CrossNodRunner = '.\remote_tools\mmfi_action_targets\Run-MMFiCrossNod.ps1'
$CrossNodOut = 'U:\runs\mmfi_action_targets_cross_nod_s42_v2'
& $CrossNodRunner -Mode CheckOnly -Outdir $CrossNodOut
```

`U:` chỉ là ví dụ disk còn trống. Có thể dùng `C:` hoặc disk local khác; dùng
output mới dưới `runs/` với basename như trên (v2/v3 nếu đổi plan). CheckOnly
chỉ kiểm code manifest/đường dẫn, không Torch/CUDA/output.
Nếu action ở chỗ khác, thêm `-CrossAction 'PATH\data_cross.npy'` và
`-NodAction 'PATH\data_nod.npy'` vào **mọi lệnh**. Không lấy bend đổi tên file.

Action cần `.provenance.json` do downloader hiện có cung cấp. Nếu không có
sidecar nhưng bạn tự xác nhận đúng hai file action thật, dùng rõ ràng
`-ApproveActionLabels` trong Geometry/Prepare/Launch/Resume/Evaluate/Export.
Script không tự đoán nhãn action từ tên file.

1. Chạy geometry **CPU, TRAIN-only**, chưa train model:

```powershell
& $CrossNodRunner -Mode Geometry -Outdir $CrossNodOut -ApproveActionLabels
```

Xem `geometry_cross.png`, `geometry_nod.png` trong output. Đây là GT/target thật
trên TRAIN, **không có model predictions**. Chọn first/middle/last sorted poison
IDs trước khi xem kết quả. Audit kiểm tất cả13305poison poses qua sáu dose,
không chọn mẫu theo tính dễ/đẹp, không đọc TEST để chọn geometry. Degenerate/
unreachable pose làm dừng và ghi geometry_failure; không tự bỏ mẫu.

2. Chỉ sau khi đã xem và đồng ý hình học, khi GPU rảnh:

```powershell
& $CrossNodRunner -Mode Prepare -Outdir $CrossNodOut -ApproveActionLabels -ApproveGeometry
```

Prepare audit toàn bộ file CSI/GT và ordered split hashes; cần original bend
prior để khớp input-audit gốc. Sau đó thử CUDA B32 thật với8poison+24clean cho
mỗi action và bỏ weights. **Prepare không tự rơi vào training.** GPU có job
khác thì từ chối; không kill/pause hay bypass guard cũ. Chờ job khác xong.

3. Khởi động hai run serial:

```powershell
& $CrossNodRunner -Mode Launch -Outdir $CrossNodOut -ApproveActionLabels
```

Giữ terminal/máy hoạt động. Log vừa hiện terminal vừa lưu output/logs. Checkpoint
atomic mỗi epoch, có model/optimizer/RNG/config/input/source/runtime binding.
Lỗi/OOM dừng, không giảm batch/retry hoặc chạy tiếp Nod sau lỗi Cross.

Nếu ngắt run, kiểm log rồi **tự quyết định** resume:

```powershell
& $CrossNodRunner -Mode Status -Outdir $CrossNodOut
& $CrossNodRunner -Mode Resume -Outdir $CrossNodOut -ApproveActionLabels
```

Resume chỉ dùng checkpoint đúng plan. Không warm-start từ leg-rotation hoặc
motion-prior cũ. Nếu fail evaluation đã tạo orphan poses NPZ, script giữ nguyên
và từ chối overwrite; gửi log trước, không tự xóa file để chạy lại.

## Output và qualitative

Mỗi `cross/`, `nod/` có final49 checkpoint, resolved cfg, poison manifest,
results.json và `poses_full_test.npz`: GT `[N,17,3]`, target/prediction
`[6,N,17,3]`, dose grid, target joints, ordered pair IDs. Không suy skeleton từ
scalar metric. Fixed TEST0/16632/33263 giữ để vẽ cùng inputs giữa actions.

Metric full TEST: T-MPJPE/Target-PCK@.2 trên target joints của **từng action**,
scale cleanGT(5,12), Clean MPJPE/PA17khớp, requested/predicted movement, DCE
theo sáu dose và mean chỉ năm dose dương, no-trigger target controls, non-target
drift. Hai action có số khớp/độ lớn payload khác nhau: không gộp trung bình để
claim action nào dễ/mạnh hơn. Nod có movement nhỏ có thể PCK cao dù không phản
ứng; phải xem no-trigger và requested/DCE. Clean-trained+trigger chưa có.

Sau cả hai hoàn tất có `action_targets_results.csv`. Gửi lại results + pose
NPZ để plot qualitative thật. Chưa sửa manuscript/PDF hay báo hiệu quả tấn công
chỉ vì code/preflight/local synthetic tests chạy được.
