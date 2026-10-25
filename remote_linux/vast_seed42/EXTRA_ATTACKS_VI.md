# Hai attack bổ sung trên Vast: MM-Fi P1S1, seed 42

Đây là **pilot thăm dò riêng**, gồm 4 victim train mới từ đầu:
MetaFi++/WiFlow × TSBA-adapted/LF-inspired CSI control. Không sửa checkpoint,
manifest hay kết quả 8 phương pháp đã chạy. Không đưa 4 hàng này vào bảng T1
đã chốt nếu chưa cập nhật protocol và công bố khác biệt về quyền attacker.

| Phương pháp | Cách dùng CSI | Quyền attacker |
| --- | --- | --- |
| TSBA-adapted | Generator tạo trigger riêng cho từng mẫu; công thức nhân của [mã tác giả TSBA](https://github.com/yujingmarkjiang/Time_Series_Backdoor_Attack) được chuyển sang CSI/pose 3D | White-box: generator được cập nhật bằng gradient qua victim |
| LF-inspired fixed smooth CSI control | Pattern DCT tần số thấp cố định, cộng với dose rồi clip biên độ; lấy cảm hứng từ [LF](https://github.com/SCLBD/BackdoorBench/blob/main/attack/lf.py) | Data-only: không học pattern từ clean surrogate |

LF control **không phải reproduction của LF gốc**: LF gốc dùng pattern đã tối ưu
trên một mô hình sạch. Bốn victim có cùng MM-Fi protocol1-s1/split seed0,
TRAIN 133056/TEST 33264, poison rho=.1, seed42, target chân MM-Fi [2,3],
6 dose 0,.2,.4,.6,.8,1 và checkpoint cố định epoch49. MetaFi++ dùng SGD
lr=.01; WiFlow dùng AdamW lr=.0001, physical batch32 và accumulation2.
Target-PCK@.2 dùng attacker target ở hai khớp, scale từ clean GT[5,12]; DCE
lấy trung bình 5 dose dương. Một seed không có SD/CI giữa các seed.
TSBA dùng biên độ multiplicative riêng, LF dùng additive epsilon=.185; đây
không phải so sánh cùng chuẩn nhiễu hay cùng quyền attacker.

## Trên máy Vast đã có dữ liệu và source v6

Upload `mmfi_two_extra_attacks_s42_overlay_v2.zip` vào `/` bằng giao diện Vast.
Đối chiếu SHA256 mà lệnh build in ra với file vừa upload. Một lần đầu:

```bash
set -e
ZIP=/mmfi_two_extra_attacks_s42_overlay_v2.zip
DEST=/workspace/mmfi_two_extra_s42_overlay_v2
test -f "$ZIP"
test ! -e "$DEST"
sha256sum "$ZIP"
mkdir -p "$DEST"
python3 -m zipfile -e "$ZIP" "$DEST"
/workspace/venvs/mmfi/bin/python "$DEST/verify_extra_bundle.py" --root "$DEST"
```

Mã nền giữ ở `/workspace/mmfi_v6/backdoor-analog`, dữ liệu tại
`/workspace/mmfi`; ZIP nhỏ không chứa lại hai thứ này. Nếu đường dẫn trên máy
khác, truyền `--repo-dir`, `--data-home` và `--dataset-receipt` tương ứng.

**Kiểm trước khi train thật:**

```bash
R=/workspace/mmfi_two_extra_s42_overlay_v2/extra_attacks_runner.py
P=/workspace/venvs/mmfi/bin/python
$P -u "$R" --dry-run
$P -u "$R" --prepare-only
```

`--dry-run` chỉ in plan. `--prepare-only` kiểm source/dữ liệu và làm GPU smoke;
chưa chạy 50 epoch. Nếu smoke lỗi, giữ log để chẩn đoán; sau khi xử lý,
chạy lại `--prepare-only` thủ công trong cùng thư mục để chạy smoke còn thiếu.
Không tự giảm batch, đổi optimizer hay dùng một receipt của run khác để vượt
gate. Kiểm GPU trống trước khi launch: `nvidia-smi`.

**Chạy 4 victim:**

```bash
tmux new-session -d -s mmfi-two-extra-s42 \
  'exec /workspace/venvs/mmfi/bin/python -u /workspace/mmfi_two_extra_s42_overlay_v2/extra_attacks_runner.py --launch >> /workspace/mmfi_two_extra_s42_v1.log 2>&1'
tail -n 30 -F /workspace/mmfi_two_extra_s42_v1.log
```

Controller chạy tối đa **một process trên mỗi GPU**: hai LF controls trước, rồi
hai TSBA. Không tự chạy lại job lỗi. `Ctrl+C` ở màn `tail -F` chỉ ngừng xem
log. Theo dõi ngắn:

```bash
/workspace/venvs/mmfi/bin/python "$R" --status
tail -n 30 /workspace/mmfi_two_extra_s42_v1.log
nvidia-smi
```

Khi controller báo hoàn tất, kiểm từng checkpoint epoch49 và kết quả cùng
input/source binding:

```bash
/workspace/venvs/mmfi/bin/python "$R" --check-only
```

Kết quả nằm dưới `/workspace/runs/mmfi_two_extra_s42_v1`. Chỉ dùng số đo khi
`--check-only` xác nhận đủ **4/4**. Nếu source v6, dữ liệu, CUDA hay bộ nhớ trên
host thay đổi, báo lỗi/không launch; giữ nguyên run cũ để kiểm tra.
