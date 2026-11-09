# BackTime HPE-Li → MetaFi++ và WiFlow trên Vân

## Material Passport

- Yêu cầu: setup BackTime cho hai backbone trên MM-Fi; ngày 2026-11-09.
- Skill: academic-research-suite, experiment-agent; phạm vi setup/kiểm thử.
- Protocol: **P1S1, seed42, rho10%**, hai victim mới từ scratch, mỗi model50
  epochs/final49; tối đa hai worker trên RTX4090. Không chạy P3S3/PiW/defense.
- Attacker: **tái dùng frozen BackTime đã fit bằng HPE-Li**, cùng adaptive
  poison IDs và paired doses trong `source/backtime/poison.json`. Không fit
  attacker bằng MetaFi/WiFlow; đây là shared-surrogate transfer + victim
  retraining, không phải ba attacker được tối ưu riêng cho từng backbone.
- Giữ nguyên scientific source `ATKBackd`, dữ liệu và run cũ. Người dùng đã
  yêu cầu push code GitHub ngày 2026-11-09; chỉ publish code/tests/docs, không
  dataset/checkpoints. Không truy cập GPU từ Admin, kill job, cài package
  hoặc cập nhật môi trường.
- Verification: CPU/synthetic tests; chưa xác minh CUDA/full MM-Fi trên Vân.
  Checkpoint nguồn mới được kiểm tra qua output người dùng cung cấp; launcher
  sẽ kiểm tra lại hash/binding/poison/fit profile khi CheckOnly/Prepare.

## Cấu hình và control

| Victim | Optimizer/lr | Batch | Epochs |
| --- | --- | --- | --- |
| MetaFi++ | SGD .01 / momentum .9 / wd0 | physical32 | 50, final49 |
| WiFlow | AdamW 1e-4 / wd1e-4 / betas(.9,.999) / eps1e-8 / clip1 | physical32 × accumulation2, effective64 | 50, final49 |

FP32/workers0/fixedLR/ordinary MPJPE; WiFlow tail accumulation sample-weighted
và flush mỗi epoch. BN theo physical32. Không đổi architecture, không ImageNet,
không native bone loss/validation plateau/early stopping của WiFlow.

Full official P1S1/split0: TRAIN133056, TEST33264, tensor3×114×10, 17 joints,
metres. Pivot1 → joints[2,3], +Z40°×dose, doses0,.2,.4,.6,.8,1.
T-MPJPE và Target-PCK@.2 tại d=1; DCE mean năm dose dương; Clean MPJPE/PA;
no-trigger control và **clean-trained model đúng backbone + cùng trigger**.
Không gọi clean PCK là Target-PCK hoặc Target-PCK là pose-level ASR.

Hai clean references **reuse read-only**, không thêm hai clean trainings.
Phải có config/cache/final49 checkpoint và training input/poison audits hợp lệ.
Đánh giá mới replay clean utility từ cache trong .05mm. Historical clean
training input records chỉ có ordered identities, không full content hash tại
thời điểm train; không che giới hạn này. Full content hiện tại được đối chiếu
với source BackTime audit và bind vào plan mới.

## Bàn giao và chạy

Theo yêu cầu mới, bàn giao qua GitHub `toan31072006/backdoor-analog`, branch
`main`; không cần copy ZIP. Chỉ pull khi **không có worker đang dùng checkout**
và working tree sạch. Không reset/stash tự động hoặc ghi đè sửa đổi của Vân.

```powershell
$btRepo = 'C:\Users\Van\code_backtime_4090_20261106'
$btDirty = @(git -C $btRepo status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Không đọc được repo.' }
if ($btDirty.Count) { throw 'Repo có thay đổi local; giữ nguyên và gửi git status.' }
git -C $btRepo pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Pull lỗi; dừng, không reset hoặc chạy tiếp.' }
git -C $btRepo log -1 --oneline
$btScript = Join-Path $btRepo 'remote_tools\Start-MMFiBackTimeTransfer4090.ps1'
$btSource = 'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1\p1s1'
$btOut = 'U:\backdooranalog\runs\mmfi_backtime_transfer_metafi_wiflow_p1s1_s42_4090_v1'
powershell -NoProfile -ExecutionPolicy Bypass -File $btScript -Mode Inspect -SourceRun $btSource -Outdir $btOut
```

Nếu checkout cũ vẫn đang được sử dụng, clone vào một thư mục **mới** thay vì
pull checkout đang chạy. Dataset/env/checkpoints vẫn ở vị trí cũ. Không cập
nhật source sau Prepare hoặc giữa chừng một run đã bind plan/source hashes.

Dùng explicit arguments với `powershell.exe -File`, không đổi policy toàn máy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $btScript -Mode CheckOnly -SourceRun $btSource -Outdir $btOut
powershell -NoProfile -ExecutionPolicy Bypass -File $btScript -Mode Prepare -SourceRun $btSource -Outdir $btOut
```

Mặc định DataHome=`C:\Users\Van\backdooranalog`, env cũ
`envs\dose-backdoor\python.exe` (fallback `envs\mmfi-p3\Scripts\python.exe`).
Đổi env bằng `-PythonExe`, cùng path trong mọi mode. Không tự cài/nâng torch.

Mặc định hai clean folders:

```text
C:\Users\Van\backdooranalog\runs\mmfi_cross_backbones_s42_43_44_v1\metafiplusplus\seed42\clean
C:\Users\Van\backdooranalog\runs\mmfi_wiflow_a1_s42_v1\wiflow\seed42\clean
```

Inspect in rõ file nào thiếu, không Torch/CUDA/ghi output. Nếu checkpoint nằm
chỗ khác, truyền **folder chứa checkpoint.pt/config.resolved.yaml/eval_cache.json**
bằng `-MetaFiClean` / `-WiFlowClean` cho mọi lệnh. Không trỏ vào A1 hoặc HPE-Li.
CheckOnly chỉ đọc và kiểm tra hashes/full inputs, không smoke/train/GPU.
Nếu source hash/poison/config không khớp, dừng và gửi lỗi; không sửa manifest cũ.

Prepare yêu cầu GPU không có compute job khác, output volume local còn ≥10GiB.
Hai real CUDA B32 smokes dùng frozen trigger/đúng optimizer, chạy cùng lúc,
discard weights, lưu GPU gates và **dừng**. Gate không bảo đảm hội tụ/tốc độ hay
backdoor hiệu quả; không giảm batch/AMP khi OOM. Nếu gate lỗi, giữ output/logs,
không tự retry. Người dùng quyết định xử lý trước khi Prepare lại.

Chỉ khi Prepare thành công mới chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $btScript -Mode Launch -SourceRun $btSource -Outdir $btOut
```

Launch chạy foreground: giữ terminal mở và Windows không sleep. Hai worker
độc lập; worker xong train evaluate ngay. Checkpoint atomic mỗi epoch chứa
model/optimizer/Python+NumPy+Torch+CUDA RNG và bindings. Gọi lại **cùng Launch**
là explicit resume từ epoch-boundary cuối; không tự retry, không dùng best TEST.
Một worker lỗi không kill worker còn lại; controller đợi job còn lại thoát rồi
báo lỗi. Không bấm Launch thứ hai khi controller/worker cũ vẫn chạy.

Xem log ở terminal khác:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $btScript -Mode Status -SourceRun $btSource -Outdir $btOut
```

Status chỉ in file presence + tail logs; không lấy file tồn tại làm bằng chứng
job còn tiến triển. Controller in PID/ALIVE mỗi30s, worker in loss/epoch. Không
tự kill job stall; gửi log khi cần kiểm tra.

## Artifacts và cách báo cáo

```text
plan.json
gpu_gate.metafiplusplus.json / gpu_gate.wiflow.json
logs/
metafiplusplus/{checkpoint.pt,evaluation.json,per_sample.npz}
wiflow/{checkpoint.pt,evaluation.json,per_sample.npz}
summary.json / summary.csv
```

NPZ giữ clean/triggered XYZ, attacker target XYZ, true XYZ, ordered pair IDs,
sáu doses, target errors/no-trigger errors/DCE. Trước khi ghi complete, replay
scalar metrics/residuals từ NPZ và kiểm tra payload geometry. Saved float32
GT/labels có rounding <.001mm; chỉ geometry check dùng tolerance1e-6metres,
scalar metric replay vẫn1e-6mm/percentage point. Không đổi metric gốc.

Không dùng TEST để chọn lại trigger/poison/seed. Attacker HPE-Li cũ có target
PCK thấp cũng không phải lý do refit đến khi đẹp. Báo từng backbone riêng,
one-seed/shared-surrogate extension, không tự gọi đây là full T1/T6 hoặc gộp
hai backbone thành seed/SD. Paper cần khai báo attacker reuse nếu dùng kết quả.
