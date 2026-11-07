# Vẽ skeleton: 3 input × 6 dose

Mỗi hàng là cùng một TEST frame MM-Fi/P1S1. Sáu cột là d = 0, 0.2,
0.4, 0.6, 0.8, 1.0. Xám = GT, cam nét đứt = attacker target trên chân bị
tấn công, xanh = prediction thật. Ở d=0, target trùng GT nên không vẽ đè cam.
Góc nhìn, hộp tọa độ và tỉ lệ mét được giữ chung cho cả 18 panel. Mỗi hàng
chỉ trừ một origin lấy từ GT, không căn chỉnh prediction theo dose.

## Chạy từ GitHub trên máy Van/RTX4090

Clone vào thư mục riêng, không pull/sửa source của job đang chạy:

```powershell
git clone --depth 1 https://github.com/toan31072006/backdoor-analog.git "$env:USERPROFILE\MMFiDosePose3x6Git"
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\MMFiDosePose3x6Git\remote_tools\mmfi_dose_pose_3x6\Run-MMFiDosePoseFigure.ps1"
```

Python environment, checkpoint, dataset và action mặc định vẫn được đọc từ
`backdooranalog` hiện có như bên dưới. Source ưu tiên `backdooranalog\ATKBackd`
nếu tồn tại; nếu không có, dùng `ATKBackd` trong Git checkout chứa helper và
in rõ `Source selection`. Không chép source đè vào nơi đang train. Strict model
loading, SHA checkpoint và replay d=1 vẫn được giữ; source clone không tự được
coi là source train gốc. Các hash source thực tế được ghi vào metadata.
Nếu thư mục clone đã tồn tại, chọn một tên clone mới; không ghi đè source cũ.

### Nếu đã clone và gặp FileNotFoundError

Cập nhật CHỈ clone helper, không cập nhật `backdooranalog` đang chạy job:

```powershell
git -C "$env:USERPROFILE\MMFiDosePose3x6Git" pull --ff-only
$DoseScript = "$env:USERPROFILE\MMFiDosePose3x6Git\remote_tools\mmfi_dose_pose_3x6\Run-MMFiDosePoseFigure.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $DoseScript -CheckOnly
```

`-CheckOnly` chỉ đọc đường dẫn, kiểm các file source/sáu file CSI-GT và SHA
checkpoint; không import Torch, dùng GPU hay ghi output. Nó in TẤT CẢ đường
dẫn thiếu thay vì dừng ở thư mục thiếu đầu tiên. PASS này chỉ là path/hash
preflight, không phải xác nhận CUDA hay kết quả model.

Nếu đường dẫn mặc định checkpoint/data/action không đúng, chỉ định vị trí
THỰC TẾ bằng `-Checkpoint`, `-DataRoot`, `-ActionNpy`; source gốc khác dùng
`-SourceRoot`. Không chọn một checkpoint khác cho tiện: SHA A1 gốc vẫn bắt buộc.
Ví dụ (thay các placeholder bằng đường dẫn có thật):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $DoseScript -CheckOnly -SourceRoot 'D:\actual\ATKBackd' -Checkpoint 'D:\actual\seed42\a1\checkpoint.pt' -DataRoot 'D:\actual\Compress' -ActionNpy 'D:\actual\data_bend.npy'
```

Sau `DOSE_POSE_PATH_CHECK PASS`, chạy cùng lệnh nhưng bỏ `-CheckOnly` để
inference và vẽ. Nếu còn `MISSING`, gửi nguyên output; không tự retrain/move
dataset, vô hiệu kiểm SHA hay dừng các job khác. NPZ có sẵn có thể plot-only
mà không cần các đường dẫn export này.

## Cách offline thay thế

Chép `MMFiDosePose3x6.zip` vào Downloads của máy Van, rồi chạy PowerShell:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\MMFiDosePose3x6.zip" -DestinationPath "$env:USERPROFILE\Downloads\MMFiDosePose3x6"+powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\MMFiDosePose3x6\Run-MMFiDosePoseFigure.ps1"
```

Không cần train lại. Không tự cài thư viện, tải dataset, sửa checkpoint hoặc
dừng process. Dùng Python hiện có:
`%USERPROFILE%\backdooranalog\envs\dose-backdoor\python.exe`.

Các đường dẫn mặc định (có thể chỉ định rõ bằng các tham số trên):

- Source: `%USERPROFILE%\backdooranalog\ATKBackd`
- Checkpoint: `%USERPROFILE%\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1\seed42\a1\checkpoint.pt`
- Data: `%USERPROFILE%\backdooranalog\datasets\Compress`
- Reference action: `%USERPROFILE%\backdooranalog\actions\data_bend.npy`

Exporter cũ giữ nguyên guard phối hợp với `Run-MMFiBackTime`. Nếu báo guard
đang chạy, không kill job: đợi một slot đã phối hợp rồi chạy lại. Không có
automatic retry. Batch inference chỉ gồm 3 frame mỗi dose.

## Kết quả

- Prediction NPZ: `%USERPROFILE%\Downloads\mmfi_dose_pose_s42.npz`
- Hình: `%USERPROFILE%\Downloads\MMFiDosePoseFigure_s42\fig_same_input_dose_s42.pdf`
- Preview: cùng thư mục, đuôi `.png`
- Số liệu 18 panel: cùng thư mục, đuôi `.csv` (số liệu từng frame, KHÔNG phải trung bình toàn TEST)
- Provenance và hashes: cùng thư mục, đuôi `.json`

NPZ có sẵn thì runner chỉ validate và vẽ, không inference lại. Output hình có
sẵn thì dừng để tránh ghi đè; dùng `-OutDir` mới nếu muốn vẽ lại.

Chỉ export nếu chưa có NPZ (không cần Matplotlib ở bước này):

```powershell
& "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe" -B -u "$env:USERPROFILE\Downloads\MMFiDosePose3x6\Export-MMFiDosePose.py"
```

Nếu máy Van thiếu Matplotlib, không cần cài thêm để export. Chuyển NPZ về máy
có NumPy/Matplotlib, rồi chạy plotter (không cần Torch/GPU/source/dataset):

```powershell
python .\Plot-MMFiDosePose.py --input .\mmfi_dose_pose_s42.npz --out-dir .\MMFiDosePoseFigure_s42
```

## Kiểm tra và giới hạn

Các TEST index 0, 16632, 33263 đã cố định trước khi xem prediction; không chọn
frame theo lỗi thấp. Exporter kiểm SHA checkpoint, fingerprint, epoch49, 6
dose và replay d=1 so với errors đã lưu (sai khác tối đa 0.1 mm). Plotter
kiểm lại shapes, topology, dose, hình học target và replay endpoint. Nó không
sửa prediction hoặc suy ra prediction từ target/metric. Các checks này không
phải chữ ký xác thực; cần giữ NPZ gốc và log export để truy xuất nguồn.

`test_dose_pose_plot.py` dùng dữ liệu tự tạo chỉ để kiểm tra code/layout, không
phải kết quả thực nghiệm. Preview test có watermark SYNTHETIC và không được
chèn vào bài. Cho đến khi nhận NPZ thật và xem hình, figure thực nghiệm vẫn
chưa hoàn tất. Không có chỉnh sửa tự động vào `main.tex`.

Caption gợi ý sau khi kiểm tra NPZ thật:

> Predictions of the same WiFiDoor victim on three fixed MM-Fi/P1S1 test inputs
> across six doses (HPE-Li, seed 42, rho=10%). Gray denotes ground truth,
> dashed orange the dose-matched attacker target limb, and blue the model
> prediction. Each row uses one identical CSI frame; only the trigger dose
> changes. All panels share the camera and metric scale, without prediction
> alignment. At zero dose the target equals ground truth. The three examples
> are qualitative illustrations, not full-test averages.

Hình rộng hai cột (`figure*`, `width=\textwidth`). Khi người dùng đồng ý chèn
hình thật vào bài, dùng label mới `fig:same-input-dose`; không thay các đồ thị
định lượng hiện có bằng ba ví dụ này.
