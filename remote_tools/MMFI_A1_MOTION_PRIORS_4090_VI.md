# A1 motion-prior sensitivity — máy Vân 4090

Pilot bổ sung, chỉ A1 / HPE-Li / MM-Fi P1S1 / seed42. Mặc định train hai
victim mới: **cross → nod, tuần tự một job trên cuda:0**. Đây không phải ba
target-action scenarios, không phải toàn bộ T1, và chưa có kết quả thực nghiệm.
Mỗi prior chỉ thay chuyển động nguồn sinh trigger, không thay attacker target.

## Chưa chạy đồng thời với BadNets

Log người dùng cung cấp cho thấy WiFlow/BadNets đang train. Con số GPU-util
thấp hơn 100% không chứng minh có đủ VRAM hay chạy thêm sẽ nhanh hơn. Helper này
chạy **exclusive**, không có tùy chọn bỏ qua GPU guard: hãy đợi queue BadNets
WiFlow → MetaFi++ hoàn tất trước khi Prepare/Launch. `AuditOnly` chỉ CPU/đĩa,
không dùng CUDA, nên có thể kiểm tra file trước (vẫn có thể tranh chấp I/O).
Không sửa source hoặc pull
trong `code_badnets_mmfi` khi queue đó đang chạy vì audit bind source hashes.
Không stop/kill process, không thay manifest/gate của run hiện có.

## Lấy code riêng, không sửa checkout đang train

Chạy trên **PowerShell máy Vân**, không phải máy 3090 hoặc Linux:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code_motion_priors'
    if (Test-Path "$Repo\.git") {
        if (git -C $Repo status --porcelain) { throw 'Checkout có thay đổi; dừng.' }
        git -C $Repo pull --ff-only
    } else {
        git clone --depth 1 --branch main https://github.com/toan31072006/backdoor-analog.git $Repo
    }
    if ($LASTEXITCODE -ne 0) { throw 'Lấy code lỗi; chưa Prepare.' }
}
```

Helper mặc định dùng data đầy đủ và Python có sẵn:

- DataHome: `C:\Users\Van\backdooranalog`.
- Python: `C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe`.
- Output mới: `runs\mmfi_p1s1_hpeli_a1_motion_priors_s42_4090_v1`.
- Action files mới: `actions\data_cross.npy`, `actions\data_nod.npy`.
  `actions\data_bend.npy` gốc vẫn cần cho receipt dữ liệu MM-Fi hiện có;
  mặc định không train lại victim bend.

Không cài hay thay environment, không tải dataset/checkpoint, không tự thêm
Clean/BadNets/victim ngoài những prior được chọn, không thu gọn batch hay retry.

## Tải cross/nod chính thức — không cần gdown

Nguồn là [website dự án PSBA](https://qichenzheng.github.io/psba-website/) và
[Google Drive folder do dự án cung cấp](https://drive.google.com/drive/folders/1ZdUKi9FpC3q-5CZmOx5lXOMyOEIcFJTU).
Đã đối chiếu file ID, byte count và header NumPy công khai ngày 2026-10-30:

| Prior | File nguồn | Bytes | Shape NumPy |
| --- | --- | ---: | --- |
| cross | [data_cross.npy](https://drive.google.com/file/d/1irgTzLOMkpfOQjNHI9omf56iJIeHq3dk/view) | 225720128 | (1254,3,300,25,2) |
| nod | [data_nod.npy](https://drive.google.com/file/d/1aGItQm8dSMk0FweZQnAdX7IuAaKHQx6A/view) | 237600128 | (1320,3,300,25,2) |

Hai file tổng khoảng **441.9 MiB**. Mapping nguồn cần giữ nguyên:
**bendSideways** (nghiêng thân sang bên), **crossHandsFront** (bắt chéo hai tay
phía trước), **nodHead** (gật đầu). Không gọi bend là cúi người về trước hay
cross là bắt chéo chân nếu source không nói vậy. Không đổi tên bend thành
cross/nod, không thay bằng random/synthetic motion.

Chạy khối dưới trên Vân. Nó dùng Python/NumPy có sẵn, không cài gdown hay
package, không dùng GPU, không khởi chạy training. Có thể tải trước trong lúc
BadNets train, nhưng mạng/CPU/đĩa vẫn chia sẻ tài nguyên với job đang chạy:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code_motion_priors'
    $Py = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
    & $Py "$Repo\remote_tools\Get-PSBAMotionPriors.py" --outdir 'C:\Users\Van\backdooranalog\actions' --priors cross nod --download
    if ($LASTEXITCODE -ne 0) { throw 'Tải/audit action lỗi; chưa Prepare hay Launch.' }
}
```

Downloader kiểm tra size, shape, float32 non-pickle/finite, tính SHA256 local và
lưu `data_*.npy.provenance.json` gắn file ID/URL/nhãn chính thức. **SHA256 này là
hash đo sau tải, không phải checksum do tác giả công bố**; chưa tìm thấy author
SHA256 để đối chiếu độc lập. HTTPS và schema/hash audit không chứng minh mọi
instance đúng nhãn; semantic provenance ở đây dựa trên mapping file chính thức.
Downloader không ghi đè file action có sẵn chưa rõ nguồn, không tự duyệt một file
chỉ vì tên đúng. Nếu gặp file có sẵn thiếu provenance, dừng để kiểm tra thay vì
đổi tên/xóa file hay thêm xác nhận nhãn tùy tiện.

File phải là NumPy numeric không pickle, NTU skeleton đúng shape/layout và hữu
hạn, có motion hợp lệ để chọn instance. Audit cấu trúc không chứng minh nhãn
động tác: tên file `data_cross.npy` không tự chứng minh đó là cross. Cần giữ URL
nguồn, action mapping/label hoặc kiểm tra nội dung tương ứng trước khi duyệt.
Với file tự upload ngoài downloader, `-ApproveActionLabels` là xác nhận người
dùng **đã kiểm chứng nhãn**, không phải
bypass kiểm tra kích thước, finite values, hash, provenance hay GPU guard.
Nếu mỗi cross/nod có sidecar `data_cross.npy.provenance.json` hoặc
`data_nod.npy.provenance.json` chứa `semantic_approved:true`, `action` đúng
`cross`/`nod` và `sha256` khớp file, không cần xác nhận thủ công. Không tự viết
`semantic_approved:true` chỉ để qua gate; nó ghi nhận nguồn gốc nhãn đã kiểm tra.
Downloader chính thức tự tạo sidecar nên các lệnh mặc định dưới đây không cần
`-ApproveActionLabels`. Chỉ dùng switch đó cho file riêng có nguồn/nhãn đã được
kiểm chứng độc lập.

Kiểm tra file đã upload trên Vân:

```powershell
Get-Item 'C:\Users\Van\backdooranalog\actions\data_cross.npy','C:\Users\Van\backdooranalog\actions\data_nod.npy' |
    Select-Object FullName,Length
Get-FileHash 'C:\Users\Van\backdooranalog\actions\data_cross.npy','C:\Users\Van\backdooranalog\actions\data_nod.npy' -Algorithm SHA256
```

Sau tải thành công, có thể audit CPU trước trong lúc BadNets còn chạy:

```powershell
$S = 'C:\Users\Van\code_motion_priors\remote_tools\Start-MMFiA1MotionPriors4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode AuditOnly
```

`AUDITED_ONLY` không cho phép Launch: vẫn cần Prepare/CUDA B32 smoke lúc GPU rảnh.

## Prepare riêng, Launch riêng — Giữ SSH mở

Sau khi queue BadNets kết thúc và nhãn hai action đã được kiểm chứng:

```powershell
$S = 'C:\Users\Van\code_motion_priors\remote_tools\Start-MMFiA1MotionPriors4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

Chỉ khi Prepare kết thúc thành công với `PREPARED_ONLY`, kiểm tra receipts rồi
chạy riêng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Launch foreground, không phải tmux/ScheduledTask/WMI. **Giữ SSH mở** trong lúc
train. Mở SSH thứ hai để xem Status; process còn sống không phải bằng chứng
training còn tiến triển:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Nếu file ở nơi khác, truyền `-CrossAction 'D:\actions\data_cross.npy'` và
`-NodAction 'D:\actions\data_nod.npy'` giống nhau ở mọi phase. Không chuyển
file/đổi tên/hash sau Prepare. Các tham số `-Priors` và action paths cũng phải
giữ nguyên giữa Prepare/Launch/Status/Export. Chạy từ PowerShell qua `& $S`
khi cần array (`& $S -Mode Prepare -Priors @('bend','cross','nod') ...`);
Windows PowerShell `-File` không phải giao diện truyền array nhiều phần tử.
Mặc định không cần `-Priors` vì đã là cross,nod.

## Recipe được giữ cố định

| Thành phần | Giá trị |
| --- | --- |
| Dataset/victim | MM-Fi P1S1, HPE-Li, full TRAIN 133056 / TEST 33264 |
| Seed/lịch | seed42; từ scratch; 50 epochs/final49; không chọn best TEST |
| Victim train | SGD lr0.001, momentum0.9, weight decay0; B32; workers0 |
| Attack | A1 native multicarrier; eps0.185; rho0.1; uniform; paired doses |
| Target | pivot1 → [2,3], Z40*d; không đổi target theo prior |
| Evaluation | d=0,0.2,0.4,0.6,0.8,1; cùng official TEST |

Giữ operator, target, poison indices/doses và training recipe; chỉ đổi motion
prior. Đây là native-amplitude sensitivity, không claim equal-noise: thay prior
có thể làm thay đổi perturbation thực tế, vì vậy giữ distortion diagnostics.

Bảng nhỏ gồm T-MPJPE@1 ↓, Target-PCK@0.2@1 ↑, mean-positive DCE ↓, Clean MPJPE ↓,
Clean PA-MPJPE ↓; lưu same-victim no-trigger control và đủ six-dose artifacts.
DCE chỉ trung bình năm dose dương; Target-PCK không phải ASR. Không dùng Clean
all-joint để thay target-joint baseline. Clean-trained reference/controls chưa
có matching artifact phải ghi unavailable, không tự train thêm Clean.

Nếu thêm bend, đây là một victim mới ở cùng output/manifest chỉ khi đã chọn
ngay từ Prepare. Không chép số bend lịch sử làm một hàng đối chứng mặc định:
chỉ reuse khi có audit khớp đầy đủ source/config/data/target/poison/recipe.
Nếu muốn bảng **ba victim mới** thay vì hai mặc định, tải đủ ba prior qua
`Get-PSBAMotionPriors.py --outdir ... --priors bend cross nod --download`, rồi
dùng `-Priors @('bend','cross','nod')` qua `& $S` cho mọi phase ngay từ AuditOnly.
Ví dụ chỉ Prepare sau khi GPU rảnh:

```powershell
& $S -Mode Prepare -Priors @('bend','cross','nod')
```

Đừng thêm bend vào manifest cross/nod đã freeze; chọn bộ prior trước AuditOnly
và dùng output mới nếu đổi bộ. Mặc định hai victim cross/nod vẫn để bend
historical **not included**, không tạo hàng bend giả để đủ ba hàng.

Một seed là pilot; chưa có mean +/- độ lệch chuẩn nhiều seed và không đủ cho
claim ổn định đa seed. Báo đủ kết quả cross/nod, kể cả nếu kém.

Local tests không thay cho audit full dữ liệu, CUDA B32 smoke trên Vân hoặc
full training/evaluation. Prepare không phải bằng chứng efficacy.
