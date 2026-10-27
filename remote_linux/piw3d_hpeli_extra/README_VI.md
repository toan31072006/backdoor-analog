# PiW3D HPE-Li: Smooth-LF-CSI và TSBA adapted, seed 42

Run này huấn luyện hai victim HPE-Li độc lập trên Person-in-WiFi-3D: `smooth_lf_csi` và `tsba_adapted`. Mỗi victim dùng 200 epoch (cuối là epoch 199), batch 32, AdamW `lr=0.001`, cùng train/test, poison identities, dose và target hai khớp chân của run HPE-Li v3 đã hoàn tất. Clean v3 chỉ được tải để làm đối chứng khi đánh giá; không dùng để khởi tạo victim. Đây là **thử nghiệm thích nghi một seed**, không phải bảng đầy đủ của bài báo gốc.

## Chạy trên máy Vân, một RTX 4090

Đây là cách chạy chính. Trên máy Vân, từ checkout GitHub mới nhất, mở PowerShell:

```powershell
$S = "$env:USERPROFILE\code\remote_tools\Start-PiWHPELiTwoExtra4090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

`Prepare` tự xuất đối chứng Clean/A1 v3 tại chỗ nếu chưa có, kiểm tra toàn bộ dữ liệu và chạy **hai GPU smoke đồng thời** trên chính RTX 4090. Chỉ khi bước đó đạt mới chạy `Launch`. Controller chạy 2 job trên `cuda:0`, mỗi job giữ physical batch 32. Nó từ chối GPU đang có compute process khác và không tự giảm batch, không tự chạy lại. Không cần chuyển dữ liệu hay checkpoint sang Vast.

Theo dõi bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = "$env:USERPROFILE\backdooranalog\runs\piw_hpeli_two_extra_s42_4090_v1"
Get-ChildItem "$R\logs\train\*.log" -ErrorAction SilentlyContinue |
    ForEach-Object { "===== $($_.Name) ====="; Get-Content $_.FullName -Tail 5 }
```

Output mới nằm dưới `runs\piw_hpeli_two_extra_s42_4090_v1`; run v3 gốc vẫn giữ nguyên. Nếu controller chết sau khi tạo lock, `-Mode UnlockStale` chỉ lưu lock cũ khi PID đã thoát và GPU rảnh. Resume đọc checkpoint cùng plan sau khi gate được kiểm tra lại.

## Tùy chọn: chạy trên Vast 2×5090

Phần dưới chỉ dùng nếu sau này chuyển thí nghiệm sang Vast; máy Vân không cần những bước chuyển ZIP này.

## 1. Xuất đối chứng trên máy Vân

Dùng đúng môi trường Python và checkout đã tạo run v3. Trong PowerShell:

```powershell
$Py = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor\python.exe"
$Export = "$env:USERPROFILE\code\remote_tools\Export-PiWHPELiV3CleanReference.py"
& $Py $Export
if ($LASTEXITCODE -ne 0) { throw 'Xuất đối chứng lỗi' }
```

Exporter đọc run `piw_hpeli_body_axis_s42_lr001_v3`, kiểm tra checkpoint cuối và kết quả đã lưu, rồi tạo `Downloads\piw_hpeli_v3_clean_reference.zip`. Ghi lại SHA256 in ra. ZIP chỉ chứa artifact đối chứng; không chứa dataset. Nếu checkout trên máy Vân chưa có exporter, lấy file từ commit GitHub của run này trước khi chạy.

Chuyển ZIP bằng máy Windows có cả file và SSH key sang Vast, vào đúng đường dẫn:

`/workspace/piw3d/references/piw_hpeli_v3_clean_reference.zip`

Sau chuyển, đối chiếu SHA256 ở hai máy (`Get-FileHash ... -Algorithm SHA256` và `sha256sum ...`). Không dùng ZIP checkpoint của MetaFi++ hay WiFlow thay thế.

## 2. Chuẩn bị trên Vast

Trong terminal Vast, từ checkout của repo này:

```bash
cd /workspace/backdoor-analog
bash remote_linux/piw3d_hpeli_extra/start.sh --dry-run
bash remote_linux/piw3d_hpeli_extra/start.sh --prepare-only
```

`--dry-run` chỉ in plan. `--prepare-only` xác minh toàn bộ dữ liệu và ZIP đối chứng, rồi thử batch thật trên hai GPU; không train. Chờ các job MetaFi++/WiFlow hiện tại kết thúc trước `--prepare-only`, vì smoke cần GPU rảnh. Cần dataset ở `/workspace/piw3d/datasets/Person-in-WiFi-3D_processed_v1` và action ở `/workspace/piw3d/actions/data_bend.npy`. Nếu checkout nằm ở vị trí khác, chỉ cần `cd` vào root repo đó.

## 3. Chạy và xem tiến độ

Sau khi prepare thành công, GPU rảnh và bạn quyết định bắt đầu:

```bash
bash remote_linux/piw3d_hpeli_extra/start.sh --launch
bash remote_linux/piw3d_hpeli_extra/start.sh --status
tail -n 30 /workspace/runs/piw3d_hpeli_two_extra_s42_v1/controller.log
```

Launcher dùng tmux tách khỏi phiên SSH: TSBA trên `cuda:0`, Smooth-LF trên `cuda:1`, mỗi GPU một job. Log từng cell nằm dưới `.../logs/train/` rồi `.../logs/evaluate/`. Controller đánh giá toàn bộ TEST sau khi train đạt epoch 199; không tự retry, không chọn checkpoint bằng TEST. Khi xong, `controller_complete.json` và `hpeli/seed42/<method>/evaluation.json` được tạo. `--status` chỉ cho biết file hiện có; xem log và kiểm tra artifact để xác nhận kết quả.

Mọi thay đổi source, dataset, action hoặc ZIP sau `--prepare-only` làm gate mất hiệu lực. Nếu cần sửa thí nghiệm, dùng thư mục output mới. `--unlock-stale` chỉ dùng khi controller đã chết và lock còn sót; nó xác minh PID không còn và GPU rảnh trước khi lưu lock cũ.
