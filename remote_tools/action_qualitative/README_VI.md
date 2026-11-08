# Qualitative HPE-Li: sáu trường hợp, chỉ plot CPU

## Material Passport

- Yêu cầu 2026-11-09: push code lên GitHub; người dùng pull/chạy trên Vân rồi chỉ lấy figure về.
- Dữ liệu: prediction XYZ thật đã lưu, seed 42, rho 10%; không train/evaluate lại.
- Sáu trường hợp: MM-Fi/P1S1 và Person-in-WiFi-3D, mỗi dataset có Bend prior / leg rotation, Cross v3, Nod.
- Quan trọng: “Bend” lịch sử là tên **trigger motion prior**, payload xoay chân. Không được gọi đó là một torso-bending target đã train.
- Không sửa nguồn ATKBackd, checkpoint, nhãn, NPZ, JSON evaluation hoặc manuscript. Đây là trình bày exploratory một seed, không phải full T7/anatomical validation.

## Chạy trên Vân

```powershell
Set-Location -LiteralPath 'C:\Users\Van\PiWMetaFiTwoExtra4090'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; stop here' }
powershell -NoProfile -ExecutionPolicy Bypass -File '.\remote_tools\action_qualitative\Run-ActionQualitative.ps1'
```

Không chạy lại training launcher. Script chỉ import NumPy/Matplotlib, không Torch/CUDA.
Không tự cài package/thay env. Nếu thiếu NumPy/Matplotlib thì dừng và báo lỗi.

Mặc định đọc:

- Cross/Nod: `U:\backdooranalog\runs\cross_nod_mmfi_piw_s42_2jobs_v3\runs\<mmfi|piw>_action_targets_cross_nod_s42_v3\<cross|nod>\poses_full_test.npz`, kèm `results.json`.
- MM-Fi Bend: `C:\Users\Van\Downloads\mmfi_dose_pose_s42.npz`, file export đã dùng để vẽ figure dose trước đó, không phải raw action skeleton `data_bend.npy`.
- PiW Bend: `C:\Users\Van\backdooranalog\runs\piw_hpeli_body_axis_s42_lr001_v3\predictions.npz`, kèm `evaluation.json`. Nếu run local không còn, tự đọc hai member cần thiết trong `references\piw_hpeli_v3_clean_reference.zip`; không giải nén/load checkpoint.

Nếu U không có, dùng `DataHome\runs`. Có thể chỉ rõ `-ActionRoot`, `-MMFiBendNpz`, `-PiWBendRoot`, `-PiWBendArchive`, `-Python`, `-Outdir`.
Ví dụ đổi chỗ file MM-Fi export:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File '.\remote_tools\action_qualitative\Run-ActionQualitative.ps1' -MMFiBendNpz 'D:\my_exports\mmfi_dose_pose_s42.npz'
```

Preflight yêu cầu đủ sáu trường hợp trước khi ghi figure. Nếu thiếu file, dừng và in đường dẫn thiếu; không dựng prediction giả, không tự chạy GPU để lấp chỗ thiếu.
Không ghi đè output có sẵn: chọn `-Outdir` mới cho lần chạy tiếp.

## Hình xuất ra

Mặc định: `U:\backdooranalog\figure_exports\actions_hpeli_s42_v1` (hoặc `DataHome\figure_exports` nếu không có U).

- `six_cases_qualitative_real.png/.pdf`: hình tổng hợp 2 dataset × 3 action, mỗi trường hợp có no-trigger và triggered ở cùng frame TEST 0.
- `<mmfi|piw>_<bend|cross|nod>_qualitative_real.png/.pdf`: sáu figure chi tiết; mỗi figure có 3 frame TEST cố định, dose 0 / 0.6 / 1 và zoom vùng target. Khớp target lớn hơn, giữ drift của các khớp khác.
- `selected_frame_metrics.csv`: số của đúng những frame đang vẽ, không thay bảng toàn TEST.
- `provenance.json`: ID, nguồn, SHA256, quy tắc chọn mẫu, giới hạn và checksum output.
- ZIP bên cạnh thư mục output chứa figure và metadata, **không** chứa NPZ/checkpoint/dataset.

MM-Fi lấy TEST `[0,16632,33263]`, PiW `[0,1293,2585]`: first/middle/last, không chọn theo điểm tốt.
Kiểm tra ID và GT giống nhau giữa ba target trong mỗi dataset trước khi so hình.
Một hệ tọa độ rigid từ clean GT được áp dụng cho tất cả pose của cùng ví dụ; **không** align/recenter/scale prediction riêng.
Nod nhìn nghiêng; hai target còn lại nhìn chính diện xiên 25 độ. Projection orthographic chỉ để hiển thị; CSV vẫn tính từ XYZ 3D gốc.
Giữ cả failure cases (đặc biệt MM-Fi Nod); không giả vờ cả sáu trường hợp thành công.

## Lấy ZIP về máy mình

Chạy trong PowerShell **máy local** sau khi Vân in `QUALITATIVE_6_CASES_OK`:

```powershell
scp windows:U:/backdooranalog/figure_exports/actions_hpeli_s42_v1.zip 'E:\wifi\actions_hpeli_s42_v1.zip'
```

Nếu ZIP local đã có, chọn tên mới để giữ bản cũ. Khi remote dùng C hoặc Outdir khác thì thay đường dẫn theo `FIGURES_ZIP`.

## Kiểm thử

```powershell
python -B -m pytest remote_tools/action_qualitative/test_action_qualitative.py -q -p no:cacheprovider
```

Các fixture trong test là synthetic để kiểm tra code, không đưa vào figure thực nghiệm.
Local Nod full NPZ dùng để kiểm tra renderer thật; toàn bộ sáu file trên Vân và hình cuối cần chạy/kiểm tra thực tế sau pull.
