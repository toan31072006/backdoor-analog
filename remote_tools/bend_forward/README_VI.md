# Bend cúi thân về trước — HPE-Li, seed 42, hai dataset

Đây là **A1/WiFiDoor**, không phải BackTime. Trigger vẫn dùng motion prior
`actions/data_bend.npy` lịch sử; nhãn đầu ra MỚI là cúi thân, không phải xoay chân.
Không đổi tên, ghi đè hay dùng trọng số của các run leg-rotation trước đây.

| Dataset | Nhãn Bend mới | Recipe giữ từ action-target runner |
| --- | --- | --- |
| MM-Fi/P1S1 | Gốc chậu 0 cố định; xoay 7–16 quanh trục qua chậu, hướng `y4-y1` (hông trái trừ phải) | SGD .001, momentum .9, wd 0, batch 32, workers 4, 50 epoch/final49 |
| PiW one-person | Xoay [0,1,2,3,4,6,8,10] quanh đường nối hai hông 5–7; hông/chân không đổi | AdamW .001, wd .01, batch 32, workers 0, drop_last=True, 200 epoch/final199 |

Góc cộng thêm là `40*d` độ, không ép mọi tư thế thành một action class. Hai xương
vai–hông PiW được giữ độ dài vì CẢ HAI hông nằm trên trục xoay. Không thêm khớp
ảo vào label. Chiều trước là quy ước `cross(leftHip-rightHip, up)`; cần xem hình
TRAIN thực trước khi duyệt. Hình chiếu cạnh dùng cùng body frame của pose sạch
cho cả ba dose; bên phải hình là chiều trước đã khai báo, body-up không phải
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
$bendHelper = 'U:\backdooranalog\code\BendForward_20261111\remote_tools\bend_forward\Run-BendForward.ps1'
& $bendHelper -Mode CheckOnly
& $bendHelper -Mode Geometry
```

Geometry chỉ đọc TRAIN labels/action prior, kiểm tra TOÀN BỘ mẫu poison và sáu
dose, xuất hình 3 ví dụ first/middle/last cố định; không train hay dùng TEST để
thiết kế. Xem hai hình:

```text
U:\backdooranalog\runs\mmfi_bend_forward_s42_v1\geometry_bend_forward.png
U:\backdooranalog\runs\piw_bend_forward_s42_v1\geometry_bend_forward.png
```

Chỉ sau khi đồng ý chiều cúi, lấy **PID hiện tại** của WiFlow/BackTime transfer:

```powershell
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'run_(a1_wiflow|piw_backtime_three_backbones|backtime_transfer[^\s]*)\.py'
} | Format-List ProcessId, CommandLine
# Thay số dưới bằng PID vừa kiểm tra; KHÔNG dùng lại PID từ log cũ.
$bendPeerPid = 12345
& $bendHelper -Mode Prepare -ApproveGeometry -PeerPid $bendPeerPid
& $bendHelper -Mode Launch -PeerPid $bendPeerPid
```

Prepare full input hash + real TRAIN B32 CUDA smoke, bỏ trọng số smoke. Chỉ chấp
nhận RTX4090/24GiB, GPU PID là worker của mình hoặc đúng peer chỉ định. Peer có
thể kết thúc bình thường; vẫn dùng PID đã lưu. Không cho PID khác thế chỗ, không
kill peer, không giảm batch/AMP/retry. Nếu driver không liệt kê GPU PIDs hoặc
smoke OOM: dừng, gửi log; không bypass guard. Một smoke không bảo đảm mọi batch
sau này vừa VRAM. Không thay source/recipe/plan sau Geometry.

Launch chạy MM-Fi rồi PiW **tuần tự**, chỉ thêm một worker so với WiFlow.
Terminal này cần mở; không chạy thêm helper thứ hai. Checkpoint mỗi epoch có
model/optimizer/RNG/plan/source/input/execution binding. Dừng ở epoch đã lưu
thì dùng `-Mode Resume -PeerPid $bendPeerPid`, không dùng Launch. Không tự retry
và không xoá file `.saving`/lock/artifact để vượt mismatch.

Mỗi dataset chỉ mở TEST sau fixed-final; lưu full GT/targets/predictions/IDs ở
`bend_forward/poses_full_test.npz`, T-MPJPE/T-PCK, requested/predicted movement,
DCE từng dose và mean năm dose dương, sạch MPJPE/PA, no-trigger control,
triggered-GT error và non-target drift. `Status` chỉ báo có file, không chứng
nhận checkpoint hợp lệ. Eval lỗi sau khi đã ghi một phần giữ nguyên orphan
artifacts để kiểm tra, không ghi đè tự động.
