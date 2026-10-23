# WiPose 2D: pilot HPE-Li Clean + A1, seed 42

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: implementation + local validation; chưa launch trên máy Vân.
- Origin Date: 2026-10-23, Asia/Bangkok.
- Verification Status: LOCAL_VALIDATION_NOT_FULL_WIPOSE_RESULT.
- Version Label: wipose_hpeli_a1_s42_v1.
- User chọn HPE-Li Clean + A1 trên WiPose và xác nhận máy Vân chưa có dữ liệu.
  Đây là nhánh độc lập, không sửa hay xóa run PiW/MM-Fi/WiFlow đã có.
- User xác nhận commit/push riêng bộ WiPose sau khi vượt test local; không
  đồng nghĩa đã launch trên Vân hoặc đã có kết quả WiPose thật.
- Chuẩn bị code/dữ liệu không tự cấp quyền thuê Vast, launch job khác, hoặc push
  thay đổi ngoài bộ WiPose đã được người dùng duyệt.

### Gate local ngày 2026-10-23

**373 passed**, exit0: 199 test WiPose (47 data/preparation, 75 adapter/metrics,
51 runner/resume/replay, 26 helper) và 174 test hồi quy PiW. HPE-Li WiPose thật
đã qua một bước SGD trên CPU với input synthetic batch32: output `[32,1,18,2]`,
gradient finite, parameters được cập nhật. Review độc lập không thấy lỗi chặn.
Đây không phải full WiPose data/CUDA run hay kết quả chứng minh attack hiệu quả;
schema/counts/ordered lists trên Vân vẫn phải qua preflight trước khi launch.
Không sửa source khoa học hoặc kết quả PiW/MM-Fi/WiFlow lịch sử.

## Đây là nhánh nào?

Đây là **HPE-Li scratch kế thừa data/model contract của nhánh WiPose trong
DT-Pose**, rồi áp A1 adaptation cho pose **2D/18 khớp**. Không phải full pipeline
pretraining + pose decoding DT-Pose; không gọi WiPose là dataset pose 3D.

- Input `[3,90,5]`: ba nhóm 30-bin trong mỗi channel, global min-max theo frame.
- TRAIN giữ frame hiện tại nếu frame tiếp theo có trong cùng danh sách; TEST
  dùng hết danh sách, không next-frame filter. Không tự re-split hay lấy subset.
  Runner kiểm raw-list counts 132847/33753 theo dataset gốc. Tất cả frame được
  kiểm shape/finite/GT reference(5,8)>0 trước GPU; lỗi không bị âm thầm loại bỏ.
- Hai victim độc lập từ scratch, thứ tự **Clean → A1**; A1 không fine-tune Clean.
- Seed 42; 50 epoch/final49; batch32; SGD lr0.01, momentum0.9, weight_decay0;
  ordinary MPJPE; fixed LR; FP32; workers0; TRAIN `drop_last=True` như nhánh nguồn.
  Không AMP, pretrained, early stopping hoặc chọn checkpoint bằng TEST.
- A1: rho0.1; uniform poison selection, paired trigger/label dose U(0.2,1),
  epsilon0.185; action prior thật `actions/data_bend.npy`, không random fallback.
  Trigger operator native áp lên ba nhóm amplitude 30-bin, không normalize lần hai.
- Payload trong mặt phẳng tọa độ ảnh: pivot right hip8, targets right knee9/right
  ankle10; góc40°×dose, convention toán học dương trên x/y được cấp. Không tự thêm Z.
  Geometry được khóa trước TEST; không điều chỉnh để vượt kết quả PiW chưa đạt.
- Một seed/two victims là pilot, không phải full comparison/defense/ablation hoặc
  bằng chứng ổn định qua nhiều seed.

Nguồn: [Wi-Pose chính thức](https://github.com/NjtechCVLab/Wi-PoseDataset),
[DT-Pose](https://github.com/cseeyangchen/DT-Pose),
[process.py nhánh WiPose](https://github.com/cseeyangchen/DT-Pose/blob/main/data/wipose/process.py).
Source/hash cụ thể được ghi trong provenance local; không gọi phiên bản `main`
trên web hôm nay là phiên bản gốc bất biến.

## 1. Tải dữ liệu chính thức trên Vân

Dùng **Anaconda PowerShell**, không CMD; không dán kèm dấu nhắc `PS C:\...>`.
Link Google Drive dưới đây do repo dataset chính thức công bố. Không phải link
PiW cũ `1tr98...` và không tải vào folder PiW.

```powershell
$WIP_HOME = "$env:USERPROFILE\backdooranalog"
$WIP_ARCHIVE = "$WIP_HOME\downloads\Wi-Pose.rar"
New-Item -ItemType Directory -Force "$WIP_HOME\downloads" | Out-Null

# gdown chỉ cài trong base, không đổi torch/numpy của env thí nghiệm đã khóa.
conda run --no-capture-output -n base python -m pip install gdown
if ($LASTEXITCODE -ne 0) { throw 'Cài gdown lỗi; dừng tại đây.' }

conda run --no-capture-output -n base python -m gdown `
  'https://drive.google.com/uc?id=1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9' `
  --continue -O "$WIP_ARCHIVE"
if ($LASTEXITCODE -ne 0) { throw 'Tải lỗi; chưa giải nén.' }

Get-Item -LiteralPath $WIP_ARCHIVE | Select-Object FullName,Length
Get-FileHash -LiteralPath $WIP_ARCHIVE -Algorithm SHA256
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free
```

Không coi SHA local là checksum được tác giả chứng nhận: nó chỉ giúp nhận diện
file đã nhận. Chưa khẳng định dung lượng giải nén hoặc số byte trước khi xem file
thực tế; cần đủ chỗ cho archive, raw, processed và checkpoint cùng tồn tại.

Kiểm tra có 7-Zip và xem cấu trúc **trước** khi giải nén. `Expand-Archive` dành
cho ZIP, không dùng cho RAR.

```powershell
$WIP_7Z = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
if (!$WIP_7Z) {
    $WIP_7Z = Join-Path $env:ProgramFiles '7-Zip\7z.exe'
}
if (!(Test-Path -LiteralPath $WIP_7Z -PathType Leaf)) {
    throw 'Chưa tìm thấy 7-Zip. Cài/cấu hình 7-Zip rồi xem archive; chưa giải nén.'
}
& $WIP_7Z l "$WIP_ARCHIVE"
if ($LASTEXITCODE -ne 0) { throw 'Không đọc được archive; chưa giải nén.' }
```

Nếu danh mục có absolute paths, `..`, hoặc cấu trúc không rõ thì dừng và gửi
output. Không đoán archive chứa ngay `Train`/`Test` ở root. Sau khi xác nhận danh
mục, giải nén vào **folder staging mới**, không trộn vào PiW hay folder đã có:

```powershell
$WIP_RAW_STAGING = "$WIP_HOME\datasets\WiPose_raw_v1"
if (Test-Path -LiteralPath $WIP_RAW_STAGING) {
    throw 'Folder staging đã tồn tại; giữ nguyên, kiểm tra trước khi giải nén tiếp.'
}
& $WIP_7Z x "$WIP_ARCHIVE" "-o$WIP_RAW_STAGING" -aos
if ($LASTEXITCODE -ne 0) { throw 'Giải nén chưa thành công; giữ file để kiểm tra.' }

Get-ChildItem -LiteralPath $WIP_RAW_STAGING -Directory
Get-ChildItem -LiteralPath $WIP_RAW_STAGING -Filter Train -Directory -Recurse |
  Select-Object FullName
Get-ChildItem -LiteralPath $WIP_RAW_STAGING -Filter Test -Directory -Recurse |
  Select-Object FullName
```

Đặt `$WIP_RAW` thành **folder cha thật của cả `Train` và `Test`** từ output trên.
Không chạy prepare khi còn đang giải nén. Không dùng thư mục raw làm output.

## 2. Chuẩn bị MAT → NPZ, không ghi đè raw

Hai schema phải phân biệt rõ:

- `--schema dtpose`: MAT có `CSI`/`SkeletonPoints`; giữ active transpose/reshape
  của process.py. Hàm DWT có trong source nhưng **không được gọi ở active branch**;
  tên output `*_Amplitude_DWT` không chứng minh đã denoise DWT.
- `--schema official`: MAT công khai có `csi_serial`/`jointsVector`; lấy amplitude
  `abs` từ CSI rồi transpose/reshape, giữ x/y và bỏ confidence. Đây là conversion
  profile công khai, **không claim input bytes giống bản DT-Pose preprocessed**.
- Mặc định `auto` chỉ nhận schema đã biết, đồng nhất; không trộn hai schema.
  Tọa độ là annotation ảnh ×0.001 trong loader, không phải đơn vị metre.

```powershell
cd "$env:USERPROFILE\code"
$WIP_PY = "$WIP_HOME\envs\dose-backdoor\python.exe"
$WIP_ROOT = "$WIP_HOME\datasets\WiPose_processed_v1"

# Sửa đúng theo cấu trúc đã in, không đoán thêm hoặc bỏ một lớp folder.
$WIP_RAW = "$WIP_RAW_STAGING"
if (!(Test-Path "$WIP_RAW\Train") -or !(Test-Path "$WIP_RAW\Test")) {
    throw 'WIP_RAW chưa phải folder cha của Train và Test; chỉnh đường dẫn trước.'
}

& $WIP_PY -u .\ATKBackd\prepare_wipose.py `
  --raw-root "$WIP_RAW" --outdir "$WIP_ROOT" --schema auto
if ($LASTEXITCODE -ne 0) { throw 'Prepare chưa đạt; giữ output để kiểm tra, chưa launch.' }
```

Nếu thiếu `Train_data_list.txt`/`Test_data_list.txt`, ưu tiên giữ/lấy đúng ordered
lists của dữ liệu. Nếu không có, cần chọn **explicit `--generate-lists`** để dùng
tất cả file MAT theo thứ tự sorted trong từng split đã cấp. Đây là thay đổi thứ
tự so với `os.listdir` của tác giả và được ghi trong `preparation.json`, không
claim seed-order reproduction nguyên bản. Chỉ thêm flag sau khi xác nhận split;
không tự di chuyển mẫu giữa TRAIN/TEST. Khi prepare bị ngắt giữa chừng, không
xóa output hoặc ghi đè. Có thể tiếp tục explicit với đúng đường dẫn/schema/flags:

```powershell
& $WIP_PY -u .\ATKBackd\prepare_wipose.py `
  --raw-root "$WIP_RAW" --outdir "$WIP_ROOT" --schema auto --resume
```

Nếu lượt đầu có `--generate-lists` thì lượt resume cũng phải có flag đó. Resume
chỉ đọc/so sánh các NPZ đã có rồi bỏ qua file hợp lệ; raw/list/source/schema thay
đổi hoặc file `.saving` dang dở bị từ chối để kiểm tra, không tự xóa/ghi đè.
Không thêm `--resume` cho folder dữ liệu khác không có preparation binding.

MAT v7.3 cần `mat73`; nếu thiếu, cài trước khi tạo manifest thí nghiệm và ghi lại
package. Không tự cập nhật torch/numpy của experiment env. Preparer không tải,
giải nén, denoise, đổi split hoặc sửa file raw.

Processed layout sau COMPLETE:

```text
WiPose_processed_v1/
  preparation.json
  Train_data_list.txt
  Test_data_list.txt
  Train_Amplitude_DWT/<identity>.npz
  Test_Amplitude_DWT/<identity>.npz
```

Tên folder kế thừa contract, không phải tuyên bố đã áp DWT. Runner rehash dữ
liệu/ordered identities; số TRAIN sau next-frame filter có thể khác số MAT gốc.

## 3. Preflight rồi chạy đúng một worker

Code mới phải có trên Vân trước khi gọi helper. Không bỏ qua lỗi git collision;
không `reset --hard`, `git clean` hay xóa file untracked/run để ép pull.

```powershell
cd "$env:USERPROFILE\code"
$env:PYTHONUTF8 = '1'
Test-Path .\remote_tools\Start-WiPoseHPELiTask.ps1
Test-Path "$WIP_HOME\actions\data_bend.npy"

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiTask.ps1 -DryRun

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiTask.ps1 -PreflightOnly
```

DryRun là metadata/source/runtime, không đọc full raw, dùng GPU hay ghi output.
PreflightOnly kiểm dữ liệu/model/trigger trên CPU, không train hoặc đăng ký Task.
Phải hoàn tất prepare và có action prior trước full Clean+A1 preflight. CleanOnly
là tùy chọn explicit, không được báo CleanOnly là kết quả A1.

```powershell
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\remote_tools\Start-WiPoseHPELiTask.ps1
```

Mặc định task `WiPose-HPELi-Seed42`, output
`runs/wipose_hpeli_a1_s42_v1`, manifest `wipose_hpeli.resolved.json`. Clean rồi A1
chạy nối tiếp, chỉ **một worker GPU tại một thời điểm**. Không chia hai victim
cạnh tranh cùng GPU trong pilot này; helper không dừng job khác.

Chỉ muốn Clean trước: thêm `-CleanOnly`, task mặc định riêng
`WiPose-HPELi-Clean-Seed42`, cùng output. Khi Clean xong, chạy helper không flag
để xác thực/reuse Clean rồi train A1 từ scratch; không overwrite checkpoint Clean.
Không đổi dữ liệu/source/runtime/recipe giữa hai lần.

Helper chỉ đăng ký sau dry-run và preflight; có thể phải đợi hashing trước khi
thấy Task/cell log. Không launch lặp vì chưa thấy log. Task dùng current SID,
S4U/Limited/Hidden; task cùng tên nhưng khác account/config không bị ghi đè.
Root/Clean/A1 locks được probe và recheck sau preflight, không xóa lock file.
Không tự retry OOM, giảm batch, đổi recipe hoặc kill process khác.
Ngắt SSH không phải lệnh dừng Task, nhưng sleep/tắt máy, mất điện hay kill process
vẫn ngắt công việc; không có bảo đảm sống qua những tình huống đó.

## 4. Log, audit và metric

```powershell
$WIP_OUT = "$env:USERPROFILE\backdooranalog\runs\wipose_hpeli_a1_s42_v1"
Get-ScheduledTask -TaskName WiPose-HPELi-Seed42 -ErrorAction SilentlyContinue |
  Select-Object TaskName,State

foreach ($kind in @('stdout','stderr','wrapper-error')) {
  Get-ChildItem "$WIP_OUT.task.*.$kind.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 20 }
}
Get-ChildItem "$WIP_OUT\hpeli\seed42\*\console.log" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
  ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 20 }

& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_wipose_hpeli.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$WIP_OUT" --status
```

`Running` không chứng minh tiến triển; final epoch49 chưa đủ khi TEST chưa chấm.
Sau khi log báo complete, audit rồi mới đọc report:

```powershell
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -u `
  .\ATKBackd\run_wipose_hpeli.py `
  --data-home "$env:USERPROFILE\backdooranalog" --outdir "$WIP_OUT" --check-only
if ($LASTEXITCODE -ne 0) { throw 'Audit chưa đạt; gửi log, chưa train lại.' }
Get-Content "$WIP_OUT\wipose_pilot.md" -Encoding UTF8
```

- Clean MPJPE/PA-MPJPE trên 18 khớp, đơn vị **tọa độ ảnh gốc (pixel nếu annotation
  được cấp là pixel)**; nhân1000 chỉ đảo numeric scaling của loader, không phải
  phép đổi metre→mm. Không gộp với các số mm của MM-Fi/PiW.
- T-MPJPE và Target-PCK chỉ ở khớp9/10 so với attacker target; không align. PCK
  dùng 0.2×clean-GT distance(5,8), không scale từ prediction hoặc đổi sau TEST.
  Target-PCK là tỷ lệ khớp đạt ngưỡng, không gọi là pose-level ASR.
- Đủ dose0/.2/.4/.6/.8/1; DCE là sai số của dịch chuyển predicted so với requested,
  summary chỉ mean năm dose dương. Lưu same-victim no-trigger và Clean+trigger
  controls, StillDCE và drift ngoài target; không diễn giải PCK cao riêng lẻ là
  attack thành công như bài học từ pilot PiW.

Giữ kết quả kể cả attack chưa đạt. TEST không dùng chọn target/threshold/recipe;
không chuyển dataset rồi loại hồ sơ PiW khỏi audit. Local synthetic/CPU và Task
mocks chỉ xác nhận implementation, không thay full WiPose/CUDA run hoặc efficacy.
