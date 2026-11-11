# Bend cúi thân về trước — HPE-Li, seed 42, hai dataset

## Bản chạy v3 — lần đầu train Bend, sửa guard WDDM

BackTime đã xong theo xác nhận của người dùng; Bend CHƯA train. V2 bị chặn
ngay tại guard trước Prepare/smoke/training, không phải một victim cần Resume.
V3 giữ nguyên TOÀN BỘ payload đã duyệt (MM-Fi v1 +40d, PiW v2 -40d), trigger,
seed42, poison10%, optimizer, lịch 50/200 epoch và metric. Không chạy lại Clean
hay BackTime. Output/checkout v1 và v2 giữ nguyên; không pull/sửa source của chúng.

Guard đối chiếu GPU0 XML + PID/tên/đường dẫn hiện tại của Windows. Chỉ nhóm
desktop có tên VÀ đường dẫn bảo vệ đã liệt kê, với type G/C+G trong WDDM, mới
được chấp nhận. System PID4 có ngoại lệ kernel rất hẹp. Không bỏ qua mọi C+G,
không cho Python khác qua, không kill process hay reset driver. C+G vẫn có
compute context: đây là cho phép desktop cùng GPU, KHÔNG phải bảo đảm GPU độc
quyền, không chứng thực publisher/file và không bảo đảm VRAM cho toàn bộ run.

`GpuCheck` chỉ đọc, không tạo plan/output/lock hay cấp phát CUDA. `RunApproved`
đòi `-ApproveGeometry`, source SHA đúng operator đã duyệt và output v3 CHƯA
tồn tại. Nó chạy GpuCheck -> CheckOnly cả hai -> Geometry TRAIN cả hai ->
Prepare cả hai (hash đầy đủ + B32 smoke) -> Launch MM-Fi rồi PiW. Phần Geometry
được làm lại để khóa hash mới; KHÔNG thay label hay train thêm model. Nếu bất
kỳ bước nào lỗi, dừng hẳn, không tự retry/resume/ghi đè; gửi log trước khi chạy tiếp.
Kết quả kiểm thử và giới hạn: `VALIDATION_V3.md`.

## Bản sửa v2 — chiều cúi PiW

PiW v1 xoay ngược chiều cúi trong ba ví dụ TRAIN đã xem. v2 sửa **góc xoay 3D
thật sang `-40*d` độ** quanh cùng trục hông `y5-y7`; không chỉ lật hình.
MM-Fi giữ nguyên payload v1 và `+40*d`. Chiều trước PiW được khai báo là
`-cross(leftHip-rightHip, up)`, cố định theo dataset, không đoán dấu từ từng
đầu gối. Đây là hiệu chỉnh từ TRAIN, không phải kết luận về handedness hay
chứng minh giải phẫu cho mọi pose. Đối chiếu cũ và giới hạn xem
`VALIDATION_V2.md`; `VALIDATION.md` là hồ sơ kiểm thử v1 lịch sử.

Phần này là lịch sử sửa chiều v2: hình v2 của CẢ HAI dataset đã được người dùng
duyệt. Helper hiện tại mặc định `RunVersion=3`, từ chối output v1/v2. Chỉ chuyển
tiếp phê duyệt cho đúng source payload đã khóa SHA, không cho thay target âm thầm.

Đây là **A1/WiFiDoor**, không phải BackTime. Trigger vẫn dùng motion prior
`actions/data_bend.npy` lịch sử; nhãn đầu ra MỚI là cúi thân, không phải xoay chân.
Không đổi tên, ghi đè hay dùng trọng số của các run leg-rotation trước đây.

| Dataset | Nhãn Bend mới | Recipe giữ từ action-target runner |
| --- | --- | --- |
| MM-Fi/P1S1 | Gốc chậu 0 cố định; xoay 7–16 quanh trục qua chậu, hướng `y4-y1` (hông trái trừ phải) | SGD .001, momentum .9, wd 0, batch 32, workers 4, 50 epoch/final49 |
| PiW one-person | Xoay [0,1,2,3,4,6,8,10] quanh đường nối hai hông 5–7; hông/chân không đổi | AdamW .001, wd .01, batch 32, workers 0, drop_last=True, 200 epoch/final199 |

Góc có dấu là MM-Fi `+40*d`, PiW `-40*d` độ, không ép mọi tư thế thành một action class. Hai xương
vai–hông PiW được giữ độ dài vì CẢ HAI hông nằm trên trục xoay. Không thêm khớp
ảo vào label. Chiều trước là quy ước theo dataset ở trên; cần xem hình
TRAIN thực trước khi duyệt. Hình chiếu cạnh dùng cùng body frame của pose sạch
cho cả ba dose; bên phải hình là chiều trước đã khai báo. Body-up lấy từ torso
sạch chiếu vuông góc trục hông, ĐỘC LẬP với dấu forward nên không bị lật khi
đổi chiều trước; body-up không phải
ước lượng trọng lực. Bảo toàn xương không chứng minh tính giải phẫu,
thăng bằng, tiếp xúc chân hay nhận dạng action.

Full MM-Fi TRAIN133056/TEST33264, PiW TRAIN27876/TEST2586; poison10%,
native A1 epsilon .185, fixed paired U(.2,1), sáu dose0,.2,.4,.6,.8,1.
Mỗi dataset train đúng một victim từ scratch, ordinary all-joint MPJPE.
Không train lại Clean, không warm-start từ Clean. Clean lịch sử vẫn giữ nguyên;
chỉ ghép làm đối chứng sau audit riêng. Hiện report đánh dấu control chưa được
cung cấp; không phải full T7/comparison paper. T-PCK không phải clean PCK/ASR.

## Trên máy Vân

Clone bản code mới vào `U:\backdooranalog\code\...`, **không pull vào checkout
đang chạy WiFlow/BackTime**. Không dừng process cũ. Temp/output đều ở U vì C đầy.
Helper mặc định dùng `C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    New-Item -ItemType Directory -Path 'U:\backdooranalog\setup_tmp' -Force | Out-Null
    $env:TEMP = 'U:\backdooranalog\setup_tmp'
    $env:TMP = $env:TEMP
    $bendRepo = 'U:\backdooranalog\code\BendForwardWDDMV3_20261112'
    if (Test-Path -LiteralPath $bendRepo) { throw 'Checkout exists; preserve it, do not overwrite/pull.' }
    git clone --depth 1 --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git $bendRepo
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed; stop here.' }
    $bendHelper = Join-Path $bendRepo 'remote_tools\bend_forward\Run-BendForward.ps1'
    & $bendHelper -Mode RunApproved -ApproveGeometry -PeerPid 0
}
```

Geometry chỉ đọc TRAIN labels/action prior, kiểm tra TOÀN BỘ mẫu poison và sáu
dose, xuất hình 3 ví dụ first/middle/last cố định; không train hay dùng TEST để
thiết kế. Xem hai hình:

```text
U:\backdooranalog\runs\mmfi_bend_forward_s42_v3\geometry_bend_forward.png
U:\backdooranalog\runs\piw_bend_forward_s42_v3\geometry_bend_forward.png
```

Lệnh trên là lần chạy mới, giữ terminal mở. Không dùng lại PID BackTime14968
đã kết thúc. Nếu cần chạy từng bước thủ công hoặc Resume sau checkpoint thật,
không dùng lại `RunApproved`; trước hết kiểm tra trạng thái và log:

```powershell
$bendHelper = 'U:\backdooranalog\code\BendForwardWDDMV3_20261112\remote_tools\bend_forward\Run-BendForward.ps1'
& $bendHelper -Mode Status
# Chỉ đọc GPU, không training/ghi output:
& $bendHelper -Mode GpuCheck -PeerPid 0
```

Prepare full input hash + real TRAIN B32 CUDA smoke, bỏ trọng số smoke. Chỉ chấp
nhận RTX4090/24GiB, GPU PID là worker của mình, desktop đã đối chiếu hoặc đúng peer chỉ định. Peer có
thể kết thúc bình thường; vẫn dùng PID đã lưu. Không cho PID khác thế chỗ, không
kill peer, không giảm batch/AMP/retry. Nếu driver không liệt kê GPU PIDs hoặc
smoke OOM: dừng, gửi log; không bypass guard. Một smoke không bảo đảm mọi batch
sau này vừa VRAM. Không thay source/recipe/plan sau Geometry.

Launch chạy MM-Fi rồi PiW **tuần tự**, chỉ một worker mới.
Terminal này cần mở; không chạy thêm helper thứ hai. Checkpoint mỗi epoch có
model/optimizer/RNG/plan/source/input/execution binding. Dừng ở epoch đã lưu
thì dùng `-Mode Resume -PeerPid 0` (nếu đã Prepare với peer0), không dùng Launch. Không tự retry
và không xoá file `.saving`/lock/artifact để vượt mismatch.

Mỗi dataset chỉ mở TEST sau fixed-final; lưu full GT/targets/predictions/IDs ở
`bend_forward/poses_full_test.npz`, T-MPJPE/T-PCK, requested/predicted movement,
DCE từng dose và mean năm dose dương, sạch MPJPE/PA, no-trigger control,
triggered-GT error và non-target drift. `Status` chỉ báo có file, không chứng
nhận checkpoint hợp lệ. Eval lỗi sau khi đã ghi một phần giữ nguyên orphan
artifacts để kiểm tra, không ghi đè tự động.
