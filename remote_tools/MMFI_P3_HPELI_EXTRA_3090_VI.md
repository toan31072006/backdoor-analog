# HPE-Li: hai attack control trên MM-Fi Protocol 3, RTX 3090

Runner này train **Smooth-LF-CSI** và **TSBA adapted** từ đầu cho P3 S3/S2/S1,
seed 42: tổng cộng 6 victim mới. Từng setting chạy hai process đồng thời trên
`cuda:0` (physical batch 32); cả ba setting chạy lần lượt. Clean và A1 P3 đã
có là đối chứng riêng, không train lại hay ghi đè. Không có BadNets, SIG,
MetaFi++, WiFlow hoặc defense trong lượt này.

Đây là **thí nghiệm mở rộng một seed**, chưa phải bảng T1 nhiều seed. Smooth-LF
là control CSI cố định lấy ý tưởng LF, không phải tối ưu trigger bằng surrogate
như bản gốc. TSBA adapted cần quyền huấn luyện white-box. Hai phương pháp có
budget khác nhau; báo rõ trong bài.

Trên PowerShell máy 3090, dùng một checkout GitHub mới chứa
`remote_tools/Start-MMFiP3HPELiExtra3090.ps1`, dữ liệu MM-Fi đã chuẩn bị,
`actions/data_bend.npy`, và Python env `envs/mmfi-p3/Scripts/python.exe`:

```powershell
$S = "$Repo\remote_tools\Start-MMFiP3HPELiExtra3090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

`Prepare` chạy dry-run, đối chiếu split P3 với receipt HPE-Li trước đó nếu có,
và smoke cả hai operator với hai HPE-Li batch32 cùng lúc. Nó không train full.
Nếu báo `PREPARED_ONLY`, bắt đầu bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode LaunchDetached
```

Sau khi nối lại SSH, kiểm tra PID và log vì Windows host có thể đóng process
theo remote session. Runner giữ checkpoint sau từng epoch và có thể gọi lại
`LaunchDetached` để resume với cùng source và plan; không tự retry khi worker lỗi.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
$R = "$env:USERPROFILE\backdooranalog\runs\mmfi_p3_hpeli_extra_s42_3090_v1"
Get-ChildItem "$R\p3_*\*\console.log" -ErrorAction SilentlyContinue |
    ForEach-Object { "===== $($_.Directory.Parent.Name) $($_.Directory.Name) ====="; Get-Content $_.FullName -Tail 4 }
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*run_mmfi_p3_hpeli_extra_3090.py*'
} | Select-Object ProcessId,ParentProcessId,CommandLine
```

Sau khi sáu victim đánh giá xong:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
Get-Content "$R\p3_extra_attack_results.json" -Raw
```

Mỗi record có T-MPJPE@1, Target-PCK@0.2, Clean MPJPE, Clean PA-MPJPE,
mean-positive DCE, sáu hàng dose và paired per-sample arrays. Checkpoint chọn
cố định cuối epoch 49, không chọn bằng TEST.
