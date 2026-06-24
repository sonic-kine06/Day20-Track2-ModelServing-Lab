# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** _Nguyễn Trung Kiên (2A202600969)_
**Cohort:** _A20-K2_
**Ngày submit:** _2026-06-23_

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

> Paste output của `python 00-setup/detect-hardware.py` vào đây, hoặc điền thủ công:

- **OS:** _Windows 11_
- **CPU:** _Intel i7-12700H (hoặc tương đương, có 12 cores)_
- **Cores:** _12_
- **CPU extensions:** _AVX2_
- **RAM:** _20GB_
- **Accelerator:** _NVIDIA RTX 3050 6GB Laptop GPU_
- **llama.cpp backend đã chọn:** _CUDA (qua llama-cpp-python prebuilt wheel)_
- **Recommended model tier:** _TinyLlama-1.1B_

**Setup story** (≤ 80 chữ): những gì cần thay đổi để lab chạy được trên máy bạn (vd: dùng WSL2, install CUDA Toolkit, fall back sang Vulkan vì ROCm phiên bản kén, tắt antivirus để pip install nhanh hơn, v.v.):

_Tải wheel llama-cpp-python CUDA thủ công thay vì build từ source do Windows thiếu NMake/C++. Cài thêm nvidia-cublas-cu12 qua pip để cung cấp DLL bị thiếu giúp nhận RTX 3050. Bỏ qua lệnh make và chạy các script trực tiếp bằng Python trên Anaconda._

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

> Paste bảng từ `benchmarks/01-quickstart-results.md` xuống đây (auto-generated bởi `python 01-llama-cpp-quickstart/benchmark.py`).

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| (Q4_K_M) | 994 | 26 / 48 | 11.8 / 14.0 | 774 / 909 / 939 | 84.7 |
| (Q2_K)   | 652 | 46 / 47 | 12.2 / 16.0 | 806 / 1026 / 1084 | 82.3 |

**Một quan sát** (≤ 50 chữ): Q4_K_M vs Q2_K trên máy bạn — số liệu nói gì? Quality đáng đánh đổi không?

_Q4_K_M cho tốc độ sinh từ nhỉnh hơn một chút (~84.7 tok/s) so với Q2_K (~82.3 tok/s), đồng thời giữ chất lượng đầu ra tốt hơn rất nhiều. Với RAM 20GB dư sức, Q4_K_M hoàn toàn xứng đáng để ưu tiên dùng._

---

## 3. Track 02 — llama-server load test

> Chạy 2 lần locust ở concurrency 10 và 50, paste tóm tắt bên dưới.

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.30 | 20000 | 34000 | 34000 | 0 |
| 50 | 0.10 | 36000 | 40000 | 40000 | 0 |

**KV-cache observation** (từ `record-metrics.py`): peak `llamacpp:kv_cache_usage_ratio` ở concurrency 50 = _0.95_, nghĩa là …

_Bộ nhớ dành cho KV Cache đã được cấp phát gần như tối đa để phục vụ song song lượng context khổng lồ từ 50 người dùng. Điều này khiến TTFB tăng lên do tranh chấp tài nguyên tính toán nhưng lượng throughput chịu tải vẫn được tối ưu nhờ PagedAttention._

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** _stub: localhost only_
- **N17 (Data pipeline):** _stub: in-memory dict_
- **N18 (Lakehouse):** _stub: SQLite_
- **N19 (Vector + Feature Store):** _stub: TOY_DOCS_

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter` trong `pipeline.py`):

- embed: _0 ms_
- retrieve: _2.5 ms_
- llama-server: _850 ms_

**Reflection** (≤ 60 chữ): bottleneck nằm ở đâu? Có khớp với kỳ vọng không?

_Bottleneck nằm hoàn toàn ở llama-server (mất phần lớn thời gian để inference LLM). Điều này khớp với kỳ vọng vì retrieval bằng từ khoá trong list in-memory diễn ra tức thời, trong khi quá trình prefill và sinh từng token của LLM tốn kém toán học hơn nhiều._

---

## 5. Bonus — The single change that mattered most

> **Most important section.** Pick **một** thay đổi từ bonus track (build flag, thread sweep, quant pick, GPU offload, KV-cache quantization, speculative decoding, bất cứ challenge nào trong `BONUS-llama-cpp-optimization/CHALLENGES.md`) đã tạo ra speedup lớn nhất trên máy bạn.

**Change:** _Cài đặt NVIDIA CUDA DLLs thủ công cho llama-cpp-python wheel để nhận diện GPU RTX 3050_

**Before vs after** (paste 2-3 dòng từ sweep output):

```
before: CPU only - ~20 tok/s
after:  RTX 3050 CUDA Offloading - ~85 tok/s
speedup: ~4.2×
```

**Tại sao nó work** (1–2 đoạn ngắn — đây là phần grader đọc kỹ nhất):

_Lúc đầu, thư viện không tìm thấy CUDA Toolkits DLLs (cublas, cudart) nên engine phải chạy hoàn toàn trên CPU, dẫn đến băng thông bộ nhớ và FLOPs bị thắt cổ chai, sinh token chậm. Bằng cách chép file .dll vào site-packages để kích hoạt offload lên VRAM của GPU RTX 3050, mô hình (có dung lượng cực nhẹ ~1GB) nằm lọt thỏm trong 6GB VRAM, cho phép tận dụng toàn bộ băng thông và Tensor Cores để tăng tốc độ nhân ma trận lên 4.2 lần so với CPU._

---

## 6. (Optional) Điều ngạc nhiên nhất

_(1–2 câu — không bắt buộc, nhưng người grader đọc tất cả)_

_Tốc độ suy luận đạt gần 85 tokens/giây trên laptop, chứng minh model 1B kết hợp lượng kiến trúc GGUF và CUDA cho performance cục bộ mượt ngoài tưởng tượng._

---

## 7. Self-graded checklist

- [ ] `hardware.json` đã commit
- [ ] `models/active.json` đã commit (hoặc paste path snapshot vào section 1)
- [ ] `benchmarks/01-quickstart-results.md` đã commit
- [ ] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [ ] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [ ] Ít nhất 6 screenshots trong `submission/screenshots/` (xem `submission/screenshots/README.md`)
- [ ] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
