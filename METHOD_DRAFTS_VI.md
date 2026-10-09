# Phát triển method: sàng lọc draft trên MM-Fi, seed 42

## Material Passport

- Origin Skill: academic-research-suite (targeted literature + experiment plan).
- Origin Date: 2026-10-09.
- Verification Status: kế hoạch và code thử nghiệm; CHƯA có kết quả effectiveness.
- Version Label: method_screening_v1.
- Phạm vi: benchmark MM-Fi lưu trữ, nghiên cứu có kiểm soát; không triển khai vào hệ thống thực tế.

Mục tiêu là cải tiến từ bản nền micro-Doppler hiện tại, không làm yếu phương
pháp đối chứng và không chỉ đổi bảng đánh giá. Các biến thể dưới đây là **giả
thuyết của nhóm**, chưa thể gọi là phương pháp tốt hơn hoặc đóng góp mới đã
được xác nhận.

`Original` ở đây là bản nền **hiện tại đã sửa topology/metric và dùng gain
zero-mean**, không phải tái hiện nguyên xi mọi chi tiết của WBackdoor lịch sử.
Các sửa lỗi đó giữ nguyên trong cả đối chứng lẫn biến thể.

## 1. Năm hướng nên cân nhắc

| Hướng | Câu hỏi nghiên cứu | Ưu tiên / trạng thái |
|---|---|---|
| A. Multi-carrier có cấu trúc | Thêm một carrier trơn, độc lập với pattern gốc có giúp mô hình nhận biết dose tốt hơn không? | Đã có prototype `md_multicarrier`; chạy draft trước |
| B. Dose-conditioned code | Khi dose thay đổi, thay đổi cả hướng pattern thay vì chỉ biên độ có giảm T-MPJPE ở nhiều dose không? | Đã có prototype `md_dose_code`; ưu tiên vì sát mục tiêu analog |
| C. Chuẩn hóa năng lượng theo CSI | Mức nhiễu tương đối đồng đều giữa các mẫu có giúp giảm biến thiên hiệu quả không? | Đã có prototype `md_energy`; đánh giá cả distortion |
| D. Generator nhận CSI + dose | Học một generator nhỏ trên surrogate, rồi đóng băng để tạo poison cho victim học ERM độc lập | Chưa triển khai; tốn thêm compute, cần chứng minh transfer |
| E. Tối ưu poison bằng gradient matching | Gradient của poison có dẫn victim độc lập đến biến đổi pose mong muốn không? | Chưa triển khai; khó và chậm hơn, để sau draft |

Không mặc định “phức tạp hơn = tốt hơn”. A kiểm tra cấu trúc carrier, B kiểm
tra cách mã hóa dose, C kiểm tra chuẩn hóa budget. Nếu thay nhiều yếu tố đồng
thời sẽ không biết cải thiện đến từ đâu.

## 2. Nguồn tham khảo ưu tiên venue A*

CVPR, ICCV và NeurIPS đều được xếp A* trong **CORE2023** trên
[cổng ranking chính thức](https://portal.core.edu.au/conf-ranks/?by=all&page=1&search=&sort=arank&source=CORE2023).
Đây là ranking của venue theo một kỳ đánh giá, không phải “rank” riêng của
paper, cũng không phải Q1 của tạp chí. Danh sách này ưu tiên nhánh A*, không
tự gắn nhãn Q1 khi chưa kiểm tra năm/ngành của journal.

| Paper / venue | Code tác giả công khai | Ý tưởng có thể học và giới hạn |
|---|---|---|
| [Input-Aware Dynamic Backdoor Attack, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html) | [VinAIResearch/input-aware-backdoor-attack-release](https://github.com/VinAIResearch/input-aware-backdoor-attack-release) | Generator phụ thuộc input, diversity/cross-trigger training. Bản gốc cho attacker điều khiển training; không đồng nghĩa poison-only CSI. Tham khảo cho D. |
| [LIRA: Learnable, Imperceptible and Robust Backdoor Attacks, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.html) | [khoadoan106/backdoor_attacks](https://github.com/khoadoan106/backdoor_attacks) | Học generator có ràng buộc nhiễu cùng mô hình. Nếu giữ data-only thì chỉ tối ưu trên surrogate, không dùng loss riêng trong victim. Tham khảo cho D. |
| [FIBA, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_FIBA_Frequency-Injection_Based_Backdoor_Attack_in_Medical_Image_Analysis_CVPR_2022_paper.html) | [HazardFY/FIBA](https://github.com/HazardFY/FIBA) | Trộn Fourier amplitude trong ảnh, giữ Fourier phase. Gợi ý nghiên cứu cấu trúc phổ cho A; không phải phép nhân micro-Doppler của mình, và Fourier phase không phải channel phase CSI. |
| [Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.html) | [ethan-yi-zeng/frequency-backdoor](https://github.com/ethan-yi-zeng/frequency-backdoor) | Phân tích frequency artifacts và smooth triggers. Gợi ý carrier trơn cho A, không chứng minh carrier DCT của mình là LF reproduction. |
| [Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0126bb7ebad258bf4ffdbbac2dd787-Abstract-Conference.html) | [khoadoan106/backdoor_attacks](https://github.com/khoadoan106/backdoor_attacks) | Generator được điều kiện hóa theo target class. Liên quan tính programmable cho B/D, nhưng class rời rạc khác dose liên tục và pose phụ thuộc input. |
| [Sleeper Agent, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/79eec295a3cd5785e18c61383e7c996b-Abstract-Conference.html) | [hsouri/Sleeper-Agent](https://github.com/hsouri/Sleeper-Agent) | Tạo hidden-trigger clean-label poison bằng gradient matching trên surrogate và kiểm tra victim độc lập. Gợi ý cho E; phải đổi objective phân loại sang pose, không phải port trực tiếp. |

Phạm vi tìm kiếm: targeted mechanism review trên proceedings chính thức và
repo tác giả, kiểm tra 2026-10-09; không phải systematic review toàn bộ ngành.
Các công trình 2020–2022 được chọn làm nền cơ chế. Không khẳng định đây là
toàn bộ SOTA mới nhất. Danh sách tập trung các paper ảnh; việc transfer sang
RF/pose là giả thuyết cần kiểm chứng. AI hỗ trợ tìm nguồn, thiết kế và code;
tác giả cần tự đọc paper trước khi quyết định novelty/citation cuối cùng.

## 3. Ba prototype thực sự làm gì?

Gọi `x` là CSI amplitude MM-Fi `(3,114,10)` trong `[0,1]`; `p0` là pattern
micro-Doppler zero-mean, RMS bằng 1; `eps=0.185`; `d` là dose.

**Original giữ nguyên:**

```text
x' = clip(x * max(0, 1 + d * eps * p0), 0, 1)
```

**A — `md_multicarrier`:** tạo carrier separable DCT trơn `p1`, bỏ mean,
orthogonal hóa với `p0`, chuẩn hóa RMS bằng 1. Dùng
`p=(p0+p1)/sqrt(2)` trong đúng phép nhân gain của Original. Mặc định mode
subcarrier 3, packet 1; đây là modulation trên tensor đã xử lý, không phải
chứng minh tạo được bằng phát RF vật lý.

**B — `md_dose_code`:**

```text
angle(d) = (pi/2) * (1-d)
p(d) = cos(angle(d))*p0 + sin(angle(d))*p1
x' = clip(x * max(0, 1 + d * eps * p(d)), 0, 1)
```

RMS pattern vẫn bằng 1 trước clipping. Dose được mã hóa bằng biên độ **và**
hướng carrier. Tại `d=1` operator đúng bằng Original; tại `d=0` là identity.
Đầu vào giống nhau ở endpoint không có nghĩa kết quả dự đoán sẽ giống nhau,
vì mô hình được train với các input khác nhau tại dose trung gian.

**C — `md_energy`:**

```text
r = x*p0 - mean(x*p0)
delta = d*eps*r*||x||2/||r||2
x' = clip(x+delta, 0, 1)
```

Trường hợp norm bằng 0 được trả identity. Với trường hợp không suy biến,
`||delta||2/||x||2 = d*eps` **trước clipping**; clipping có thể làm giảm mức
nhiễu đo được. Không có cam kết L-infinity theo từng phần tử. Đây là rule
phụ thuộc input bằng công thức, **không phải neural Input-Aware/LIRA**.

Ba prototype là code mới dựa trên baseline hiện tại; không vendoring/port
nguyên code của sáu paper. Repo tác giả được dẫn để đối chiếu ý tưởng, không
được dùng làm bằng chứng “đã tái hiện chính xác paper”.

## 4. Draft so sánh công bằng và nhanh

Năm cell: `clean`, `original`, `md_multicarrier`, `md_dose_code`, `md_energy`.

- Seed train 42; cùng split P1-S1 và seed split 0.
- Cùng tối đa **20,000 train frames / 4,096 validation frames**, lấy hai
  subset không giao nhau từ **training pool gốc** bằng subset seed 0.
  Validation draft là `training_holdout`; test chính thức chưa dùng để chọn
  method. Không dựa vào pose/score để chọn mẫu và không đổi split full gốc.
- Mặc định **15 epochs**, HPELi, SGD lr `0.001`, batch 32, momentum `0.9`.
- Bốn attacked cell: cùng poison IDs, `rho=0.4`, `d ~ U(0.2,1)` và paired
  label dose; `clean` có rho 0.
- Cùng pivot 1, target knee/foot `[2,3]`, payload xoay z `40*d` độ.
- Victim chỉ học các cặp CSI/pose bằng MPJPE ordinary ERM; không thêm
  attack-specific loss, poison mask hay dose vào API victim.
- Eval sáu dose `0,0.2,...,1`; không phải train sáu model.
- Hai GPU chạy hai **cell độc lập** đồng thời, không phải chia một model DDP.
- Checkpoint mỗi epoch; chạy lại đúng outdir/config để resume.

Đây là **sàng lọc trên subset/validation**, không so số trực tiếp với bảng
50-epoch full MM-Fi trước đây. Một biến thể chưa tốt sau 15 epochs có thể
chưa hội tụ. Các frame validation draft cùng training subjects nên có thể
tương quan theo chuỗi; không xem đây là bằng chứng generalization cuối.
Không chọn lại split để tạo kết quả thuận lợi. Sau khi chốt method mới chạy
protocol full và đánh giá trên test chính thức, không dùng test để tune tiếp.

## 5. Chạy trên mica

Đầu tiên kiểm tra GPU đang dùng; chọn GPU có quyền sử dụng. Ví dụ hai GPU
vật lý 0 và 3, nếu được phép và đang đủ tài nguyên:

```bash
cd ~/backdooranalog/code
git pull --ff-only
git log -1 --oneline
nvidia-smi
tmux new -s mmfi-method-draft
```

Trong tmux, tạo đường dẫn mới rồi chạy:

```bash
cd ~/backdooranalog/code
export CUDA_VISIBLE_DEVICES=0,3
DOSE_DRAFT_OUT="$HOME/backdooranalog/runs/mmfi_method_drafts_s42_v1"
bash remote_linux/06_run_method_drafts.sh --outdir "$DOSE_DRAFT_OUT" --devices cuda:0 cuda:1 --num-workers 4 --fresh
```

Do `CUDA_VISIBLE_DEVICES=0,3`, `cuda:0` trong process là GPU vật lý 0,
`cuda:1` là GPU vật lý 3. Nếu chỉ dùng một GPU, đặt
`CUDA_VISIBLE_DEVICES=0` và `--devices cuda:0`.
Không copy nguyên đoạn `tmux new` và lệnh chạy thành một lần paste; tạo
tmux trước, rồi paste lệnh chạy vào shell trong tmux.

`Ctrl-b`, rồi `d` để detach; **không Ctrl-c** nếu muốn job tiếp tục.

Dry-run trước train (không đụng GPU/data):

```bash
bash remote_linux/06_run_method_drafts.sh --outdir "$HOME/backdooranalog/runs/mmfi_method_drafts_s42_preview" --devices cuda:0 --dry-run
```

Resume: attach tmux hoặc chạy lại đúng lệnh, **bỏ `--fresh`**. Đổi epochs,
caps, scientific config hoặc các file code được hash trong manifest phải dùng
outdir mới. Không xóa checkpoint để
ép chạy lại. Có thể đổi số worker/device khi resume.

Windows PowerShell (máy chứa data local):

```powershell
Set-Location 'U:\backdooranalog\code'
python .\ATKBackd\run_method_drafts.py --data-home 'U:\backdooranalog' --outdir 'U:\backdooranalog\runs\mmfi_method_drafts_s42_v1' --devices cuda:0 --num-workers 0 --fresh
```

## 6. Đọc kết quả và quyết định

Trong outdir có manifest, mỗi cell có `console.log`, `checkpoint.pt`,
`config.resolved.yaml`, `poison_manifest.json`, `draft_subsets.json` và
`eval_cache.json`. Report tổng hợp dùng trạng thái DRAFT, không official.
File tổng hợp: `draft_summary.csv/json/md`; nhiễu đầu vào:
`input_distortion.csv/json`. Trong summary, T-MPJPE của Clean là phép thử
gắn trigger Original lên clean victim để kiểm tra phản ứng nền. Trong file
distortion, Clean là input sạch không sửa, nên nhiễu bằng 0; hai phép đo này
có vai trò khác nhau, không dùng hàng Clean để suy ra budget của trigger.

Hash SHA256 của skeleton tham chiếu được ghi lúc train và đối chiếu trước
resume/cache reuse cũng như đo distortion. Nếu thay `data_bend.npy` tại cùng
đường dẫn, phải dùng outdir mới; report không được trộn trigger cũ và mới.

Xem cả:

1. Clean MPJPE/PA-MPJPE/PCK: khả năng pose sạch có bị ảnh hưởng không?
2. T-MPJPE tại `d=1` và trung bình năm dose dương: có tốt hơn Original
   trên nhiều mức điều khiển, hay chỉ tốt ở endpoint?
3. `E0(d)-E(d)`: so với **cùng model, cùng target dose, không trigger**.
4. Relative-L2, RMSE, empirical L-infinity, SNR: đo trên cùng CSI validation.

Relative-L2 và SNR dùng pooled energy, không trung bình per-sample ratio.
SNR vô hạn ở dose 0 được ghi JSON `null` kèm cờ vô hạn, không ghi Infinity
không chuẩn. Mức nhiễu đầu vào thấp **không tự chứng minh** không bị detector
phát hiện hay tính khả thi vật lý. Cùng eps/RMS pattern không đảm bảo cùng
input distortion sau phép nhân/clipping; C còn đổi định nghĩa budget.

Giữ toàn bộ năm hàng kể cả biến thể kém. Bước tiếp là xác nhận method có
trade-off tốt bằng full MM-Fi **50 epochs**, cùng split/seed/budget và một
evaluation protocol được đóng băng trước khi so sánh kết quả cuối. Hướng D/E
chỉ triển khai sau khi đọc kỹ paper và chốt threat model; không âm thầm tối
ưu trực tiếp victim để thắng comparison.

Chưa tự chạy thêm training trên server, chưa cập nhật paper bằng kết quả
giả, và không thay các bảng đã hoàn thành bằng số draft.

## 7. Kiểm tra code trước khi push

2026-10-09, môi trường local Windows / Python 3.13 / PyTorch 2.9.1 CPU:

- Toàn bộ `ATKBackd/tests`: **533 passed, 7 skipped**. Đặt
  `MKL_THREADING_LAYER=SEQUENTIAL` trước khi chạy pytest để tránh xung đột
  MKL/OpenMP của môi trường Windows cục bộ; không dùng bypass duplicate OpenMP.
- Smoke test thực hiện train/evaluate một batch cho cả năm cell bằng dữ liệu
  giả và model nhỏ, kiểm tra checkpoint/cache/audit/report thật và cache resume.
- CLI dry-run không cần data/CUDA; cú pháp shell launcher và Python compile
  đã kiểm tra.

Đây là kiểm tra tính đúng của code, **không phải kết quả effectiveness trên
MM-Fi**. Chưa kiểm tra CUDA training trên server trong lần phát triển này.

## 8. Phép thử tiếp theo: Multi-carrier giới hạn nhiễu đỉnh

Profile `peak_control` (`method_peak_control_v1`) chỉ train **một** cell mới:
`md_multicarrier_peak_matched`. Giữ seed 42, 15 epochs, SGD lr 0.001, rho 0.4,
20,000 train / 4,096 validation frames, subset seed 0, cùng poison IDs/doses
và payload với draft trước. Không train lại Clean hoặc Original; không nhập
cache cũ vào manifest mới, không sửa/xóa kết quả cũ.

Với cùng CSI `x`, dose `d` và `eps=0.185`, tính hai output bằng đúng operator
float32 và clipping hiện tại:

```text
x_original = Original.inject(x, d, eps)
x_multi    = MultiCarrier.inject(x, d, eps)
b          = max(abs(x_original - x))
delta      = x_multi - x
alpha      = min(1, b / max(abs(delta)))
x_peak     = x + alpha * delta
```

Nếu candidate đã nằm trong budget thì giữ output Multi-carrier nguyên vẹn.
Nếu `b=0` và cần cap thì trả input sạch; không chia cho 0. Khi đổi về float32,
tọa độ nào làm tròn vượt budget được làm tròn vào phía `x`, để **output lưu
thực tế** thỏa `max(abs(x_peak-x)) <= b` theo từng mẫu và từng dose. Dose 0
là identity. Thu nhỏ nhiễu không thay dose của payload/nhãn mục tiêu.

Đây là giới hạn trên L-infinity **sau clipping**, không phải bắt hai trigger
có cả L2 và L-infinity bằng nhau. Quy tắc thu nhỏ phụ thuộc CSI và dose,
nên không gọi là ablation thuần chỉ thay carrier. Không dùng nhãn, dự đoán
victim hoặc thống kê validation để tune `alpha`; công thức áp dụng giống nhau
trong train và evaluation. Không suy ra tính không bị phát hiện hay khả thi RF
chỉ từ các norm đầu vào.

Chạy sau khi `nvidia-smi` xác nhận GPU được phép dùng và còn đủ tài nguyên.
Ví dụ GPU vật lý 3 (chỉ cần **một GPU** cho một cell):

```bash
cd ~/backdooranalog/code
git pull --ff-only
nvidia-smi
tmux new -s mmfi-peak-control
```

Trong shell tmux, paste riêng đoạn sau:

```bash
cd ~/backdooranalog/code
CUDA_VISIBLE_DEVICES=3 bash remote_linux/06_run_method_drafts.sh --profile peak_control --outdir "$HOME/backdooranalog/runs/mmfi_peak_control_s42_v1" --devices cuda:0 --num-workers 4 --fresh
```

Chọn GPU khác bằng cách đổi số `3`; trong process vẫn là `cuda:0`. Detach bằng
`Ctrl-b`, rồi `d`. Nếu chạy lại để resume thì giữ outdir và bỏ `--fresh`.
**Không chạy vào outdir screening cũ**: source hash/profile đã thay đổi, runner
sẽ từ chối thay vì trộn protocol. Profile screening mặc định vẫn giữ đủ năm cell.

Kết quả nằm trong `draft_summary.csv/json/md` và `input_distortion.csv/json`.
Profile mới đo Original operator trên **cùng 256 mẫu validation và sáu dose**
để đối chiếu norm, không train thêm một Original victim. File distortion có
`original_relative_l2`, `original_rmse`, `original_linf`, SNR và audit từng mẫu
trong JSON. `paired_linf_violations` phải bằng 0; vượt bound sẽ báo lỗi,
không xuất một báo cáo success giả. Hash action/trigger/subset được ghi lại.

So T-MPJPE/clean metrics của candidate với hàng Original draft cũ như một
**đối chứng lịch sử đã lưu**, không ghi nó là một run mới. Trước khi gộp kiểm
tra cùng subset hashes, action hash, poison-plan hash và các config victim/
payload/budget (ngoại trừ profile và trigger). Không dùng source hash mới để
ghi đè provenance cũ. Full 50-epoch MM-Fi và paper vẫn chưa được thay đổi.

### Kiểm tra bản peak-control trước khi push

2026-10-09: toàn bộ `ATKBackd/tests` **632 passed, 7 skipped** trên CPU local;
bao gồm train/evaluate/checkpoint/cache/resume bằng dữ liệu giả cho cả profile
năm cell cũ và một cell mới. Có 48 test riêng cho giới hạn nhiễu thực tế,
trường hợp suy biến, clipping, rounding, RNG/state và factory. CLI dry-run của
`peak_control` xác nhận đúng một cell; Python compile và shell syntax qua.
Phép stress độc lập 8,960 injection không phát hiện vượt peak Original.

Test local từng bị lỗi quyền ở temp mặc định của pytest; dùng một `--basetemp`
mới trong workspace để kiểm tra, không đổi quyền hệ thống. Trạng thái code:
đã kiểm thử; **chưa có kết quả hiệu quả MM-Fi thật/CUDA cho biến thể mới**.
