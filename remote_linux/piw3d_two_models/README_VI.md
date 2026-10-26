# Person-in-WiFi-3D: MetaFi++ và WiFlow trên Vast 2×RTX 5090

Đây là thí nghiệm **khám phá seed 42** trên bộ PiW3D một người: mỗi backbone
được huấn luyện độc lập từ đầu cho Clean, A1 và sáu baseline
(`wanet_source`, `ftrojan`, `fiba`, `bpp_source`, `smooth_lf_csi`,
`tsba_adapted`). Tổng cộng 16 victim, 200 epoch, checkpoint cố định cuối
epoch 199; TEST chỉ chạy sau khi 16 victim hoàn tất. Kết quả của HPE-Li và
MM-Fi trước đây không được xem là checkpoint hay seed mới ở đây.

## Dữ liệu cần tải lên riêng

Bundle code **không chứa dataset hoặc action prior**. Đặt cây dữ liệu đã chuẩn
bị tại:

```text
/workspace/piw3d/datasets/Person-in-WiFi-3D_processed_v1/
  train_data/train_data_list.txt
  test_data/test_data_list.txt
  ...các file CSI và pose được hai list trên liệt kê...
/workspace/piw3d/actions/data_bend.npy
```

Sao chép `data_bend.npy` thật từ máy Vân; đối chiếu SHA256 của file nguồn và
file sau khi upload. Không tạo file thay thế. Preflight kiểm tra mọi mẫu một
người được liệt kê: thứ tự, shape CSI `(3,180,20)`, 14 khớp 3D, giá trị hữu
hạn, phân tách TRAIN/TEST và hash nội dung. Bộ dữ liệu dự kiến giữ 27,876
TRAIN, 2,586 TEST theo quy tắc PiW3D đang dùng. Nếu khác, dừng và kiểm tra
dữ liệu trước khi huấn luyện. Tránh đặt dữ liệu hoặc checkpoint trong thư mục
bundle code.

## Bundle code

Trên checkout phát triển, tạo ZIP bằng
`python remote_tools/Build-PiW3DTwoModelsBundle.py --output <new-code-zip>`.
Sau khi upload lên Vast, đối chiếu SHA256 do builder in ra trước khi giải
nén. Giải nén vào một **thư mục mới**, ví dụ
`/workspace/piw3d_code_smokefix_v3/backdoor-analog`; chạy `verify_bundle.py` sau đó.
Không ghi đè checkout MM-Fi hiện có.

Launcher dùng `python` đang active (ví dụ `/venv/main/bin/python`); cần PyTorch có CUDA,
NumPy, SciPy và PyYAML. Đặt `PIW_PYTHON` nếu muốn chọn venv PiW riêng. Launcher
không tự cài package hay thay recipe.

```bash
cd /workspace/piw3d_code_smokefix_v3/backdoor-analog
python3 remote_linux/piw3d_two_models/verify_bundle.py --root .
bash remote_linux/piw3d_two_models/start.sh --dry-run
bash remote_linux/piw3d_two_models/start.sh --prepare-only
```

`--prepare-only` chỉ kiểm dữ liệu/recipe và chạy **tối đa sáu process thật đồng thời**
trong ba đợt: ba job mỗi GPU ở hai đợt đầu, hai job mỗi GPU ở đợt cuối.
Mỗi phương pháp phải qua smoke với batch vật lý 32, backward và barrier cùng
lúc. Nếu OOM hay lỗi, xem log trong `smoke/<attempt>/waveN/logs/`; bộ chạy dừng,
không tự giảm batch/đổi AMP/retry. Việc kiểm hash toàn bộ dataset có thể mất
nhiều thời gian I/O; bước launch sau đó kiểm receipt, hash action, hash hai
list chính thức, source, GPU và các smoke log/barrier, không hash lại tất cả
file dữ liệu để tiết kiệm thời gian; nó kiểm lại đường dẫn, kích thước và
mtime của toàn bộ file đã audit. Nếu dữ liệu bị sửa mà cố tình giữ nguyên cả
kích thước lẫn mtime, bước kiểm nhanh này không phát hiện được; giữ dataset
bất biến sau preflight.

Khi thấy `PREPARED_ONLY` và đã xem gate, chạy lệnh **riêng**:

```bash
bash remote_linux/piw3d_two_models/start.sh --launch
tail -F /workspace/runs/piw3d_two_models_s42_3pergpu_v2_smokefix/controller.log
```

Theo dõi từng cell:

```bash
find /workspace/runs/piw3d_two_models_s42_3pergpu_v2_smokefix/logs -type f -name '*.log' \
  -exec sh -c 'echo "===== $1 ====="; tail -n 5 "$1"' _ {} \;
bash remote_linux/piw3d_two_models/start.sh --status
```

Worker lưu `plan.json` khoa học; controller lưu riêng `controller_plan.json`, `preflight_receipt.json`,
`prelaunch_gate.json`, các log, checkpoint, kết quả từng cell và
`attack_results.json` (bảng gộp 2 model × 8 victim). Dấu
`controller_complete.json` chỉ xuất hiện sau khi cả 16 train và 16 eval
cell thành công. Nếu một cell lỗi, controller dừng và giữ mọi log/checkpoint;
không gọi lại launch một cách tự động. Không dùng dấu GPU bận hoặc một dòng
`epoch 199` làm bằng chứng cho kết quả TEST hoàn chỉnh.
