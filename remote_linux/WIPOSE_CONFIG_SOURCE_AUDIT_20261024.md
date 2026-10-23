# WiPose 2D: audit cấu hình nguồn chính thức, 2026-10-24

Phạm vi: đối chiếu GitHub tác giả và bài gốc, không chạy dataset/GPU, không
đổi recipe hiện tại, không lấy điểm benchmark để chọn cấu hình. WiPose ở đây là
Wi-PoseDataset 18 khớp **2D**, không phải Person-in-WiFi-3D hay hệ thống WiPose
3D trong một bài khác. Đây là targeted source/code audit, không phải systematic
review và không xác nhận đã reproduce kết quả của các tác giả.

## Kết luận sử dụng

Run hiện tại có mô tả chính xác là **HPE-Li scratch, adapted từ nhánh WiPose
trong DT-Pose**. Nhánh này dùng SGD .01/momentum .9, batch32, 50 epochs và
ordinary MPJPE. Nó khác full DT-Pose có pretraining, khác CSI-Former, và cũng
khác main hiện tại của GitHub HPE-Li đã được cập nhật thành HPE-Li++.
Không có một bộ optimizer/lr/batch chung cho mọi bài dùng WiPose.

## Nguồn đã pin

| Nguồn tác giả | Revision kiểm tra | Phạm vi bằng chứng |
| --- | --- | --- |
| [cseeyangchen/DT-Pose](https://github.com/cseeyangchen/DT-Pose) | `2fbbef271454e6fbb95b1a0163b8cdbb96163838` | Pose YAML, pretrain YAML, active Python branches, feeder, metrics. GitHub API main cũng trả revision này ngày audit. |
| [Toandinh1/HPE-Li-ECCV2024](https://github.com/Toandinh1/HPE-Li-ECCV2024) | `854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7` | Active WiPose training script, constants, dataset reader, model, metrics. README nói rõ repo đã update HPE-Li++. |
| [NjtechCVLab/Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset) | `ff22dda5746ad9648f38d4391f01db6ceb3b1079` | GitHub recursive tree chỉ có `README.md`, `Matrix.png`: xác minh dataset/schema/split, không có training implementation ở revision này. |

Các revision lấy bằng read-only GitHub REST `repos/<owner>/<repo>/commits/main`;
file được đọc từ raw.githubusercontent.com với revision đầy đủ, không chạy code
tải về. Các line anchor dưới đây là line của file tại revision đã pin.

## 1. DT-Pose: phải tách ba nhánh

### Full DT-Pose pose decoding, không phải HPE-Li scratch

YAML đặt batch32, total_epoch50, experiment_name DT-Pose và đường dẫn pretrained
weights. `base_learning_rate=.001` không quyết định optimizer của mọi nhánh.
Trong active pretrained WiPose branch, code load checkpoint encoder, tạo
ViT_Pose_Decoder 18×2 và dùng AdamW lr .001. Top-level batch32 được dùng cho
loader; nested `train_loader.batch_size=256` không phải physical batch ở đây.
Nguồn: [pose YAML L9-L20](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/config/wipose/pose_config.yaml#L9),
[loader L42-L58](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L42),
[pretrained branch L71-L84](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L71).

### HPE-Li scratch branch mà adapter hiện tại theo

Điều kiện cần trong source là pretrained_model_path None và experiment_name
hpeli. Active branch tạo hpelinet(18,2,subcarrier90), áp dụng hpeli initialization,
SGD lr .01/momentum .9; không truyền weight_decay. Training loss là mean norm
Euclidean theo khớp; không thấy scheduler trong pose loop. 50 epochs lấy từ
YAML. Source chấm TEST mỗi epoch rồi lưu best theo MPJPE/PA/PCK; adapter hiện
tại chủ động khóa final49. Nguồn: [scratch branch L140-L148](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L140),
[loss L222-L227](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L222),
[best selection L330-L342](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L330).

Feeder dùng official ordered lists, TRAIN chỉ giữ frame có next frame, TEST
không lọc như TRAIN; input3×90×5, min-max toàn frame, pose18×2 nhân .001.
Training shuffle/drop_last True, source workers8. PCK dùng scale GT joints5,8;
pose evaluation có MPJPE/PA-MPJPE và PCK thresholds .5/.4/.3/.2/.1.
Nguồn: [feeder L21-L29 và L35-L80](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/feeder/wipose.py#L21),
[conversion L83-L87](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/feeder/wipose.py#L83),
[loader L140-L148](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/feeder/wipose.py#L140),
[PCK scale L97-L110](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/utils.py#L97).

Seed YAML42 không đồng nghĩa exact initialization42: train_pose.py gọi
setup_seed42 rồi `torch.manual_seed(init_rand_seed=0)` khi tạo loaders trước
khi tạo model. Adapter hiện tại seed42 trực tiếp, trả thêm singleton person
dimension và độc lập triển khai active attention; đây là adaptation đã khai báo,
không phải bitwise reproduction initialization tác giả.
Nguồn: [seed/loader L39-L58](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/train_pose.py#L39).

### Pretraining nếu muốn chạy full DT-Pose

YAML:400 epochs, nominal batch4096/max-device512, mask .8, base LR1.5e-4,
weight decay .05, warmup40. Active pretrain.py dùng AdamW betas(.9,.95), LR
nhân batch/256, tức nominal pre-scheduler .0024; LambdaLR warmup/cosine và
steps_per_update8. Vì vậy không gọi1.5e-4 là actual optimizer LR cố định.
Nguồn: [pretrain YAML L10-L17](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/config/wipose/pretrain_config.yaml#L10),
[optimizer/schedule L91-L93](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/pretrain.py#L91),
[update loop L106-L123](https://github.com/cseeyangchen/DT-Pose/blob/2fbbef271454e6fbb95b1a0163b8cdbb96163838/pretrain.py#L106).

## 2. GitHub HPE-Li hiện tại: HPE-Li++, không phải recipe DT-Pose

[README L2](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/README.md#L2)
ghi repo đã update cho HPE-Li++. Không gán mọi con số của current main cho
implementation ECCV2024 ban đầu.

Active `att_wipose.py`:

- Train batch64/shuffle True; val32/test1. Official Test directory lại được
  chia đôi thành validation và final test, random_state41; khác full TEST của
  adapter. [L24-L34](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L24).
- Tạo HPEWiPoseModel mới; AdamW lr .001. Momentum/weight decay không truyền
  tường minh. `num_epochs=experiment_config['epoch']` và constant.py đặt60;
  `n_epochs=20`/`n_epochs_decay=30` là tham số LambdaLR, không phải tổng20 epoch.
  [script L39-L52](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L39),
  [constant L9-L15](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/constant.py#L9).
- Loss là MSE giữa confidence×prediction và confidence×GT, chia32 trong
  training; scheduler.step mỗi epoch. Chọn best validation PCK20 rồi load
  để chấm test. [loss L77-L104](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L77),
  [selection L172-L182](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L172),
  [reload L200-L202](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L200).

Reader dùng os.listdir thay official ordered text lists, MAT fields CSI và
SkeletonPoints, reshape9×30×5 và normalize mean/std theo9 channels; coordinates
18×2 nhân .001, giữ confidence. Model forward có3 SKUnits active rồi head36;
khác adapter2 SKUnits. [reader L10-L59](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/wipose/wipose_dataset.py#L10),
[model L81-L93](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/model/HPE_Wipose.py#L81).

Metrics có MPJPE/PA-MPJPE nhân1000 khi in và PCK5/10/20/30/40/50; PCK18 dùng
GT scale5,8. Với 2D image-coordinate labels, phép nhân1000 chỉ đảo scaling
coordinate; không tự biến thành khoảng cách vật lý mm.
[metrics L244-L283](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py#L244),
[PCK function L11-L39](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/utils/eval.py#L11).

Không thấy pretraining checkpoint load trước optimization trong active entry
script; load best.pt ở cuối là evaluation. Không dùng MM-Fi dataset_lib YAML
để suy batch/split WiPose, vì entry script WiPose hardcode loaders như trên.
Không xác nhận repo current main thực thi thành công trong audit này, cũng
không thay nó vào run đang có.

## 3. CSI-Former / original Wi-Pose dataset: code và paper phải phân biệt

Dataset repo pin chỉ có README và Matrix.png, không có optimizer/training
script để xác minh active code. README xác nhận12 volunteers/12 actions,
18 image keypoints, raw CSI5×30×3×3, schema csi_serial/jointsVector/jointsMatrix,
và split132847TRAIN/33753TEST. [README L3-L7](https://github.com/NjtechCVLab/Wi-PoseDataset/blob/ff22dda5746ad9648f38d4391f01db6ceb3b1079/README.md#L3).

Bài tác giả **CSI-Former: Pay More Attention to Pose Estimation with WiFi**,
Entropy2023 25(1):20, DOI10.3390/e25010020, §4.3 mô tả Adam lr .005,
batch8, 50epochs; LR giảm một nửa mỗi10epoch trong20epoch đầu và mỗi15epoch
trong30epoch cuối; chọn epoch có minimum loss. Đây là **paper-reported**,
không phải active-code verified. [Publisher](https://www.mdpi.com/1099-4300/25/1/20),
[author article in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9858036/).

Không tìm được official CSI-Former model/training code trong dataset repository
đã kiểm tra; không tự gán thêm weight_decay, pretraining, seed, pose rescaling
hoặc checkpoint-selection partition khi chưa có bằng chứng code. Các kết quả
PCK sạch trong paper cũng không phải Target-PCK của A1.

## Hệ quả cho run hiện tại

Giữ recipe đã khóa; việc cache chính xác input FP32 hoặc gom finite-gradient
checks là thay execution có kiểm tra parity, không phải lý do chuyển optimizer,
loss, batch, epochs, skeleton, split, threshold hay metric sang một paper khác.
Nếu sau này muốn đối chiếu full DT-Pose hoặc current HPE-Li++ thì cần một
experiment riêng, đúng model/preprocessing/split/training/checkpoint selection
của nhánh đó. Note này không cấp quyền chạy thêm experiment và không báo điểm
scientific mới.
