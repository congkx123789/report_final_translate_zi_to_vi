# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỀ TÀI NGHIÊN CỨU KHOA HỌC SINH VIÊN
**Đề tài:** Nghiên cứu Kiến trúc Lai Hybrid Attention-SSM phục vụ Chuyển phong cách Văn bản và Dịch thuật Hán - Việt  
**Mã đề tài:** 3378  
**Sinh viên thực hiện:** Hà Vũ Công (MSSV: 23020014)  
**Giảng viên hướng dẫn:** TS. Nguyễn Văn Vinh  
**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Công nghệ – ĐHQGHN (UET-VNU)  
**Thời lượng khuyến nghị:** 12 – 15 phút trình bày + 10 phút trả lời phản biện Hội đồng.

---

## I. NGUYÊN TẮC VÀNG KHI LÊN BỤC BẢO VỆ

1. **Quy tắc mở đầu (30 giây đầu tiên):** Đứng thẳng, tác phong tự tin, chào kính trọng Hội đồng, đặt các cuốn báo cáo bản cứng ngay ngắn trước mặt thầy cô, giới thiệu ngắn gọn tên và đề tài rồi đi thẳng vào vấn đề.
2. **Quy tắc điều hướng bản cứng:** Mỗi khi trình bày một công thức hay sơ đồ trên slide, hãy chủ động mời thầy cô: *"Kính mời thầy cô theo dõi thêm phương trình toán học chi tiết tại Trang X trong cuốn báo cáo trước mặt ạ."* Điều này tạo ấn tượng cực kỳ chuyên nghiệp và chuẩn mực.
3. **Quy tắc 6x6:** Slide chỉ chứa từ khóa và sơ đồ. Mắt bạn nhìn về phía Hội đồng, tuyệt đối không quay lưng đọc chữ trên màn chiếu.

---

## II. KỊCH BẢN NÓI CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE SCRIPT)

### Slide 1: Trang Tiêu đề (30 giây)
* **Keyword:** UET, Hybrid Attention-SSM, Hán - Việt, 3378.
* **Lời thoại:**
  > *"Kính thưa quý thầy cô trong Hội đồng Khoa học! Em tên là Hà Vũ Công, sinh viên Khoa Công nghệ Thông tin. Dưới sự hướng dẫn tận tâm của TS. Nguyễn Văn Vinh, hôm nay em xin phép được báo cáo kết quả đề tài Nghiên cứu Khoa học Sinh viên: **'Nghiên cứu Kiến trúc Lai Hybrid Attention-SSM phục vụ Chuyển phong cách Văn bản và Dịch thuật Hán - Việt'**."*

---

### Slide 2: Nội dung Trình bày (Agenda) (30 giây)
* **Keyword:** 4 Phần: Đặt vấn đề, Kiến trúc, Thực nghiệm, Định tính & Kết luận.
* **Lời thoại:**
  > *"Bài báo cáo của em gồm 4 phần chính: Đầu tiên là Đặt vấn đề và điểm nghẽn bộ nhớ của Transformer; thứ hai là Thiết kế kiến trúc lai đề xuất; thứ ba là Các kết quả thực nghiệm và đối sánh 200,000 steps; và cuối cùng là Đánh giá định tính dịch thuật cùng định hướng tương lai."*

---

### Slide 3: Bối cảnh & Thách thức (1 phút)
* **Keyword:** Ngữ liệu cổ phong, Tiểu thuyết trường thiên, Seq2Seq câu đơn, Decoder-Only KV-Cache.
* **Lời thoại:**
  > *"Dịch thuật Hán - Việt cổ điển có đặc thù rất khác biệt: chứa lượng lớn thuật ngữ tu tiên, kiếm hiệp và văn bản thường rất dài, lên tới hàng vạn đến hàng triệu tokens. Hiện nay, công nghệ dịch máy đối mặt hai rào cản lớn: Các mô hình NMT truyền thống dùng kiến trúc Seq2Seq (Encoder-Decoder) bị bó hẹp ở cấp câu đơn ($\le 512$ tokens), dẫn tới đứt gãy đại từ xưng hô và quên nhân vật trong văn bản dài. Trong khi đó, các mô hình Decoder-Only hiện đại lại vấp phải điểm nghẽn tính toán bậc hai $O(N^2)$ và bùng nổ bộ nhớ KV-Cache. Do đó, nhu cầu cấp bách là một kiến trúc vừa giữ ngữ cảnh trường thiên vừa vận hành bền bỉ trên phần cứng cá nhân."*
* **Dẫn chứng bản cứng:** *Mục 0.1, Trang 1.*

---

### Slide 4: Điểm nghẽn $O(N^2)$ của Transformer (1 phút)
* **Keyword:** Softmax $O(N^2)$, KV-Cache bùng nổ, 16 KB/token, 30.52 GB VRAM OOM.
* **Lời thoại:**
  > *"Các mô hình hiện nay phụ thuộc hoàn toàn vào Transformer Decoder-Only. Cơ chế Softmax Attention có chi phí tính toán bậc hai $O(N^2)$. Đặc biệt trong suy luận tự hồi quy, bộ nhớ đệm KV-Cache phình to tuyến tính: mỗi token bắt buộc tốn thêm 16 KB. Khi mở rộng tới 2 triệu tokens, Transformer đòi hỏi hơn 30.5 GB VRAM chỉ riêng cho bộ nhớ đệm, dẫn tới lỗi Out-Of-Memory trên các dòng card GPU cá nhân 16GB như RTX 5060 Ti."*
* **Dẫn chứng bản cứng:** *Phương trình (6) và Bảng 1.1, Trang 4 - 5.*

---

### Slide 5: Giải pháp Đề xuất: Kiến trúc Lai Hybrid (1 phút)
* **Keyword:** Tinh hoa 2 trường phái, Linear Attention $O(N)$, Selective SSM $O(1)$.
* **Lời thoại:**
  > *"Để giải quyết nghịch lý này, đề tài đề xuất kiến trúc lai: Nhánh 1 sử dụng Linear Attention với kernel ELU+1 giúp duy trì khả năng truy hồi ngữ cảnh toàn cục với chi phí tuyến tính $O(N)$. Nhánh 2 tích hợp Selective SSM kiểu Mamba với các tham số phụ thuộc ngữ cảnh, giúp nén và lọc thông tin vào trạng thái ẩn liên tục với kích thước cố định $O(1)$."*
* **Dẫn chứng bản cứng:** *Mục 1.2 và 1.3, Trang 3 - 4.*

---

### Slide 6: Sơ đồ Khối Tổng thể Kiến trúc (1.5 phút)
* **Keyword:** 71.97M params, 8 layers, RoPE, Micro-Heads, Elevator, LM Head.
* **Lời thoại:**
  > *"Đây là sơ đồ khối hoàn chỉnh của mô hình Evolving-Head Hybrid Attention-SSM với 71.97 triệu tham số. Token đầu vào được mã hóa qua RoPE, sau đó đi qua 8 khối biến đổi lai. Ở mỗi khối, dữ liệu được phân nhánh song song: bên trái là Micro-Head Linear Attention với Causal Scan và DynamicHeadStitcher; bên phải là Selective SSM và 1D Conv Gating. Hai nhánh được pha trộn động qua Dynamic Layer Head Elevator trước khi đưa vào LM Head."*
* **Dẫn chứng bản cứng:** *Mục 2.1, Trang 12.*

---

### Slide 7: Cơ chế 1: Mã hóa Vị trí Quay RoPE (1 phút)
* **Keyword:** Xoay vector $Q, K$, Góc $m\theta$, Ngoại suy chiều dài, Tích vô hướng tương đối.
* **Lời thoại:**
  > *"Thay vì cộng thô vector vị trí tuyệt đối làm nhiễu biểu diễn từ vựng, đề tài áp dụng RoPE xoay trực tiếp vector $Q$ và $K$ trong không gian 2D. Nhờ đó, tích vô hướng giữa hai token chỉ phụ thuộc duy nhất vào khoảng cách tương đối giữa chúng, giúp mô hình có khả năng ngoại suy chiều dài ngữ cảnh hiệu quả hơn khi suy luận."*
* **Dẫn chứng bản cứng:** *Mục 2.2, Trang 12 - 13.*

---

### Slide 8: Cơ chế 2: Micro-Head Matrix Slicing & DynamicHeadStitcher (1 phút)
* **Keyword:** 8 vi-đầu, $d_k=64$, Tăng hiệu quả biểu diễn, Cửa sổ cục bộ & toàn cục.
* **Lời thoại:**
  > *"Đề tài tổ chức 8 vi-đầu chú ý kích thước $d_k=64$ ($8 \times 64 = 512$). Đi kèm với đó là DynamicHeadStitcher tự động phân luồng theo 3 cụm cửa sổ ngữ cảnh: các vi-đầu phụ trách cửa sổ cục bộ tức thời ($w=5$), tầm trung vế câu ($w=20$), và liên kết toàn cục vô hạn ($w=\infty$)."*
* **Dẫn chứng bản cứng:** *Mục 2.3, Trang 13 - 14.*

---

### Slide 9: Cơ chế 3: Causal Linear Scan theo Chunk (1 phút)
* **Keyword:** Chunk $C=64$, Nội khối song song, Liên khối tích lũy $S_t$, Chi phí $O(N)$.
* **Lời thoại:**
  > *"Thuật toán Causal Linear Scan chia chuỗi thành các khối kích thước 64. Trong nội khối, mô hình tận dụng tính song song của GPU. Giữa các khối, mô hình cập nhật ma trận tích lũy chạy $S_t \in \mathbf{R}^{d_k \times d_k}$. Nhờ vậy, độ phức tạp giảm từ $O(N^2)$ xuống $O(N)$, hoàn toàn không cần lưu giữ ma trận Softmax $N \times N$ trong VRAM."*
* **Dẫn chứng bản cứng:** *Mục 2.5, Trang 14 - 15.*

---

### Slide 10: Cơ chế 4: Selective SSM & Dynamic Head Elevator (1 phút)
* **Keyword:** $h_t \in \mathbf{R}^d$, Cổng chọn lọc $\Delta_t$, Khởi tạo thực âm $\rho(\bar{A}) < 1$, Pha trộn $\alpha_t$.
* **Lời thoại:**
  > *"Ở nhánh SSM, mô hình rời rạc hóa hệ vi phân qua ZOH. Ma trận $A$ được tham số hóa thực âm đảm bảo bán kính phổ $\rho(\bar{A}) < 1$, triệt tiêu hoàn toàn nguy cơ bùng nổ gradient. Bộ nâng tầng Dynamic Elevator tự động tính toán trọng số $\alpha_t$ cho từng token: nếu gặp từ vựng kỹ thuật hiếm sẽ ưu tiên Attention; nếu gặp hư từ liên kết sẽ tăng tỷ trọng nén của SSM."*
* **Dẫn chứng bản cứng:** *Mục 2.6, Trang 15.*

---

### Slide 11: Kỹ nghệ Dữ liệu Streaming & Tối ưu Huấn luyện trên GPU 16GB (1 phút)
* **Keyword:** 34.3M câu song ngữ, Binary Memory-Mapped Loader, BPE 40,960, Chunked Loss giảm 60% VRAM, AdamW Decoupled.
* **Lời thoại:**
  > *"Để huấn luyện thành công mô hình trên card GPU cá nhân RTX 5060 Ti 16GB, em đã triển khai 2 giải pháp kỹ nghệ then chốt: Thứ nhất, bộ dữ liệu 34.3 triệu câu được nén nhị phân và đọc bằng Binary Memory-Mapped Loader, giữ mức tiêu thụ RAM luôn dưới 400 MB. Thứ hai, với từ vựng mở rộng 40,960 tokens, ma trận Logits ở tầng cuối rất lớn; em đã hiện thực kỹ thuật Chunked Cross-Entropy Loss chia nhỏ bước tính loss theo từng khối, giúp giảm tới 60% đỉnh VRAM và triệt tiêu lỗi tràn bộ nhớ khi lan truyền ngược."*
* **Dẫn chứng bản cứng:** *Mục 3.1, Trang 16 - 17 và mã nguồn `models/chunked_loss.py`.*

---

### Slide 12: Đối sánh Huấn luyện 200,000 Steps (1.5 phút)
* **Keyword:** Best Loss 0.1794 vs 0.2017, Throughput 377.5 vs 135.2 câu/s, Hội tụ sâu.
* **Lời thoại:**
  > *"Đây là kết quả đối sánh thực tế qua 200,000 steps huấn luyện độc lập giữa Evolving Hybrid SSM và Standard Transformer Baseline có cùng siêu tham số. Về độ hội tụ, Hybrid SSM đạt Best Loss là 0.1794 (PPL = 1.20), thấp hơn so với Transformer (Best Loss 0.2017). Về tốc độ huấn luyện, Transformer đạt 377.5 câu/giây, nhanh hơn xấp xỉ 2.8 lần so với mô hình lai (135.2 câu/giây)."*
* **Dẫn chứng bản cứng:** *Mục 3.5, Trang 22 - 24, Hình 3.5.*

---

### Slide 13: Định luật Roofline & Cường độ Số học (1 phút)
* **Keyword:** Compute-Bound khi Train, Memory-Bound khi Inference, FlashAttention, L2 Cache 32MB.
* **Lời thoại:**
  > *"Để giải thích vì sao Transformer huấn luyện nhanh hơn nhưng lại tốn tài nguyên hơn khi suy luận dài, ta dùng mô hình Roofline: Khi huấn luyện song song chuỗi ngắn, Transformer đạt cường độ số học rất cao, tận dụng tốt Tensor Cores nhờ FlashAttention. Nhưng khi suy luận sinh từ từng bước, Transformer bị nghẽn băng thông DRAM. Trong khi đó, Hybrid SSM chỉ đọc đúng 1.54 MB trạng thái nhớ, nằm gọn hoàn toàn trong bộ đệm L2 Cache 32MB của GPU RTX 5060 Ti!"*
* **Dẫn chứng bản cứng:** *Mục 1.7, Trang 9 - 10 và Mục 3.5.1, Trang 24.*

---

### Slide 14: Mở rộng Cực hạn 2,000,000 Tokens (1.5 phút)
* **Keyword:** 2M tokens, Transformer OOM tại 900k, 30.52 GB vs Ổn định 542 MB.
* **Lời thoại:**
  > *"Đây là minh chứng rõ nét cho đặc tính của kiến trúc lai: Khi kiểm thử ứng suất mở rộng tới 2 triệu tokens, Transformer chạm trần 16GB VRAM tại mốc xấp xỉ 900,000 tokens và đòi hỏi hơn 30.5 GB tại 2 triệu tokens, gây lỗi quá tải bộ nhớ. Ngược lại, nhờ trạng thái nhớ hằng số $O(1)$ chỉ 1.54 MB, đồ thị VRAM của Hybrid SSM duy trì ổn định ở mức 542 MB, chỉ chiếm khoảng 3.4% VRAM của GPU!"*
* **Dẫn chứng bản cứng:** *Mục 3.6, Trang 25, Hình 3.6.*

---

### Slide 15: Tiết kiệm Bộ nhớ VRAM & Ngữ cảnh 2M Tokens (1 phút)
* **Keyword:** Tiết kiệm 72.3% VRAM, Trạng thái $O(1)$ 1.54 MB, Xóa bỏ rào cản OOM.
* **Lời thoại:**
  > *"Nhờ cơ chế cập nhật trạng thái đệ quy $O(1)$, Hybrid SSM giữ kích thước bộ đệm cố định chỉ 1.54 MB thay vì tăng tuyến tính như KV-Cache của Transformer. Điều này giúp tiết kiệm trên 72% bộ nhớ đồ họa ở 2048 tokens và vận hành ổn định ở mức 542 MB VRAM ở 2 triệu tokens mà không gặp lỗi cạn kiệt bộ nhớ."*
* **Dẫn chứng bản cứng:** *Mục 3.3, Trang 19 - 20, Hình 3.2 & 3.5.*

---

### Slide 16: Đánh giá Điểm số BLEU Đa miền (1 phút)
* **Keyword:** 41.8 BLEU Tiên hiệp, 39.2 Kiếm hiệp, Thẩm thấu phong cách cổ kính.
* **Lời thoại:**
  > *"Trên tập kiểm thử phong cách Hán - Việt cổ điển, Hybrid SSM đạt điểm chất lượng dịch thuật 41.8 BLEU, cao hơn so với Transformer chuẩn (36.4 BLEU). Mô hình tái tạo tương đối chuẩn xác các cặp đối từ vựng cổ, cấu trúc ấn quyết và ngữ điệu trang trọng."*
* **Dẫn chứng bản cứng:** *Mục 3.3.3, Trang 20 - 21, Hình 3.4.*

---

### Slide 17: Đánh giá Định tính Đối đầu Thực tế (1.5 phút)
* **Keyword:** Bảo toàn thực thể 'Lâm Động', Trung thực học thuật, Phân tích Trade-off.
* **Lời thoại:**
  > *"Về mặt định tính: Trong câu kiếm hiệp trích xuất thực tế, Hybrid SSM đã ghi nhớ chính xác danh từ riêng phức tạp 'Lâm Động' nhờ trạng thái ẩn $h_t$, trong khi Transformer bị lẫn thành đại từ 'nàng'. Tuy nhiên, một cách rất khách quan và trung thực, ở các câu ngắn có từ vựng kỹ thuật đơn lẻ (như 'siêu thị' hay 'tự chú ý'), Transformer chuẩn vẫn có độ sắc nét ngữ nghĩa cao hơn do cơ chế Softmax 1-1 không bị nén trạng thái."*
* **Dẫn chứng bản cứng:** *Mục 3.7, Trang 26, Bảng 3.4.*

---

### Slide 18: Khả năng Triển khai & Ứng dụng Thực tế (1 phút)
* **Keyword:** Recurrent $O(1)$ State, Demo Web (`demo_app.py`), CLI ngoại tuyến (`infer_cli.py`), Độc lập Cloud.
* **Lời thoại:**
  > *"Về tính thực tiễn và triển khai: Em đã đóng gói mô hình thành công cụ dịch thuật hoàn chỉnh gồm giao diện Web tương tác trực quan `demo_app.py` và công cụ dòng lệnh `infer_cli.py`. Điểm thuận lợi là người dùng có thể thử nghiệm dịch toàn bộ các bộ tiểu thuyết trường thiên ngoại tuyến (offline) ngay trên máy tính cá nhân mà không cần kết nối internet hay trả phí thuê GPU đám mây đắt đỏ, do bộ nhớ suy luận cố định chỉ 1.54 MB."*
* **Dẫn chứng bản cứng:** *Mục 3.7.2 và mã nguồn `demo_app.py`, `infer_cli.py`.*

---

### Slide 19: Đóng góp Cốt lõi của Đề tài (1 phút)
* **Keyword:** Hiện thực mô hình 71.97M, Đặc tính $O(1)$ 1.54 MB, 34.3M ngữ liệu.
* **Lời thoại:**
  > *"Tóm lại, đề tài đạt được 3 đóng góp trọng tâm: Thứ nhất, hiện thực hóa mô hình lai 71.97M tham số chạy ổn định từ đầu; thứ hai, chứng minh bằng giải tích và thực nghiệm khả năng suy luận $O(1)$ với 1.54 MB đệm nhớ, giảm thiểu nguy cơ OOM; và thứ ba, xây dựng pipeline dữ liệu streaming tối ưu cho 34.3 triệu câu song ngữ."*
* **Dẫn chứng bản cứng:** *Mục 3.8, Trang 27.*

---

### Slide 20: Hướng Phát triển trong Tương lai (30 giây)
* **Keyword:** MoE 300M+, OpenAI Triton / CUDA Kernel, Extension đọc truyện.
* **Lời thoại:**
  > *"Trong tương lai, nhóm nghiên cứu sẽ mở rộng kiến trúc bằng kỹ thuật Mixture-of-Experts (MoE) lên 300M+ tham số, biên dịch lại kernel Causal Linear Scan bằng Triton/CUDA C++ để tăng tốc độ huấn luyện, và đóng gói thành công cụ đọc dịch truyện dài hạn thời gian thực."*
* **Dẫn chứng bản cứng:** *Mục 3.9, Trang 27.*

---

### Slide 21: Lời Cảm ơn (Q&A) (30 giây)
* **Lời thoại:**
  > *"Em xin trân trọng cảm ơn quý thầy cô trong Hội đồng Khoa học và thầy hướng dẫn đã luôn đồng hành, chỉ bảo cho em trong suốt quá trình thực hiện đề tài. Em xin kính mời quý thầy cô đặt câu hỏi ạ!"*

---

## III. BỘ 10 CÂU HỎI PHẢN BIỆN "GAI GÓC" NHẤT CỦA HỘI ĐỒNG & HƯỚNG TRẢ LỜI

### Câu 1: Tại sao tốc độ huấn luyện của mô hình lai (135 câu/s) lại chậm hơn Transformer (377 câu/s) dù độ phức tạp lý thuyết là $O(N)$?
* **Cách trả lời chuẩn xác:**
  > *"Thưa thầy cô, câu hỏi này liên quan trực tiếp đến mô hình phần cứng Roofline Model ạ (Slide 13 & Mục 1.7 trong báo cáo). Khi huấn luyện song song với chuỗi vừa phải ($N \le 128$), Transformer thực hiện phép nhân ma trận dày đặc kích thước lớn và được tăng tốc tối đa bởi kernel **FlashAttention** tận dụng SRAM của GPU Blackwell. Khi đó cường độ số học $AI \gg 100 \text{ FLOP/Byte}$, mạng nằm ở vùng Compute-Bound. Ngược lại, thuật toán Causal Scan của mô hình lai thực hiện nhiều phép cập nhật ma trận vi mô $d_k=64$, dẫn đến việc mạng bị giới hạn bởi độ trễ truy xuất bộ nhớ (Memory-Bound). Tuy nhiên, đây là chi phí tính toán chỉ trả một lần khi huấn luyện (One-time Cost), đổi lại là khả năng suy luận ổn định và tiết kiệm bộ nhớ khi triển khai thực tế."*

### Câu 2: Tại sao kết quả dịch câu ngắn của Transformer lại chuẩn xác từ ngữ hơn mô hình lai?
* **Cách trả lời chuẩn xác:**
  > *"Thưa thầy cô, đây là sự đánh đổi kinh điển giữa phép chiếu chính xác (Lossless Routing) và cơ chế nén trạng thái (State Compression) ạ (Slide 17 & Mục 1.5). Transformer dùng Softmax Attention pairwise, nó 'nhìn' trực diện từ tiếng Trung sang từ tiếng Việt 1-1 nên không thể rơi từ. Còn mô hình lai phải nén thông tin câu vào ma trận $S_t$ và vector $h_t$ thông qua kernel $\phi(x) = \text{ELU}(x) + 1$. Với các câu ngắn chứa nhiều từ rời rạc, phép nén này làm mờ nhẹ độ phân giải từ cục bộ. Nhưng ưu thế của mô hình lai lại nằm ở các đoạn văn bản dài, nơi Transformer chắc chắn bị OOM và dễ bị quên nhân vật xưng hô."*

### Câu 3: Làm thế nào em đo đạc và chứng minh được mức VRAM 542 MB ở 2 triệu tokens?
* **Cách trả lời chuẩn xác:**
  > *"Thưa thầy cô, em đã thực hiện kiểm thử ứng suất bằng script `test_extreme_2m_context.py` (Slide 14 & Mục 3.6). Trong quá trình tự hồi quy, mô hình lai không hề lưu lại vector Key và Value của các bước trước vào VRAM, mà chỉ giữ đúng 2 biến đệ quy: ma trận $S_t \in \mathbf{R}^{8 \times 64 \times 64}$ và vector $h_t \in \mathbf{R}^{1024 \times 64}$. Tổng dung lượng 2 biến này trên 8 tầng là đúng **1.54 MB** theo công thức giải tích. Còn 540 MB VRAM còn lại là trọng số mô hình và bộ đệm CUDA context cố định. Vì thế từ token 1 đến token thứ 2,000,000, dung lượng VRAM là một đại lượng hằng số cố định."*

### Câu 4: Động học gradient trong nhánh Selective SSM có bị bùng nổ (Exploding) hoặc triệt tiêu (Vanishing) khi chuỗi quá dài không?
* **Cách trả lời chuẩn xác (Bật ngay Backup Slide 22):**
  > *"Kính mời thầy cô theo dõi Slide dự phòng 22 và Mục 1.6 trong báo cáo ạ. Đạo hàm chuỗi thời gian là $\frac{\partial h_t}{\partial h_k} = \prod \bar{A}_j$. Trong code hiện thực của em, ma trận liên tục $A$ được tham số hóa cố định là số thực âm: $A_n = -\exp(\theta_n) < 0$. Do $\Delta_t > 0$, phần tử rời rạc $\bar{A}_{t,n} = \exp(\Delta_t A_n)$ luôn nằm nghiêm ngặt trong khoảng $(0, 1)$. Vì bán kính phổ $\rho(\bar{A}) < 1$, gradient được ngăn ngừa nguy cơ bùng nổ. Khi cần ghi nhớ dài hạn, mạng điều khiển $\Delta_t \to 0 \implies \bar{A} \to I$, gradient được truyền dẫn ổn định ngược qua hàng vạn bước mà không bị triệt tiêu."*

### Câu 5: Thiết kế 8 vi-đầu (Micro-Head Slicing) có làm tăng số lượng tham số mô hình không?
* **Cách trả lời chuẩn xác:**
  > *"Thưa thầy cô, hoàn toàn không làm tăng thêm tham số chiếu nào ạ (Slide 8 & Mục 2.3). Chiều ẩn $d_{model} = 512$ được giữ nguyên với 8 vi-đầu $d_k = 64$ ($8 \times 64 = 512$). Ma trận chiếu $W_q, W_k, W_v$ vẫn có kích thước $512 \times 512$. Việc phân chia và định tuyến qua DynamicHeadStitcher cho phép mô hình chuyên môn hóa từng vi-đầu cho các mẫu hình ngữ nghĩa khác nhau (từ ngữ cục bộ hay toàn cục) mà vẫn duy trì tính gọn nhẹ tối đa."*

### Câu 6: Dữ liệu 34.3 triệu câu được thu thập và tiền xử lý như thế nào?
* **Cách trả lời chuẩn xác (Bật Backup Slide 24):**
  > *"Kính mời thầy cô theo dõi Slide 24 và Mục 3.1.2. Dữ liệu được trích xuất từ kho ngữ liệu song ngữ Trung - Việt đa nguồn (tiểu thuyết tiên hiệp, kiếm hiệp, hội thoại và tin tức). Em đã tiền xử lý làm sạch, lọc bỏ câu trùng lặp, câu rác và huấn luyện bộ tách từ BPE song ngữ chuyên biệt với kích thước 40,960 tokens. Toàn bộ được đóng gói nhị phân `.bin` và nạp bằng Streaming Ring Buffer để không bao giờ chiếm quá 400 MB RAM."*

### Câu 7: Bộ cắt tỉa đầu HeadPruner hoạt động thế nào và tại sao cần nó?
* **Cách trả lời chuẩn xác (Bật Backup Slide 23):**
  > *"Thưa thầy cô, khi phân bổ qua 8 vi-đầu, ở nhiều vị trí từ vựng (như dấu câu, từ nối), một số vi-đầu có đóng góp gần như bằng không. HeadPruner sử dụng một mạng nơ-ron nhỏ ước lượng điểm quan trọng $M_{\text{imp}} \in (0, 1)$ cho từng đầu và áp mặt nạ mềm/nhị phân. Kỹ thuật này giúp mô hình triệt tiêu các vi-đầu đóng góp thấp ($<0.1$), chống phân tán chú ý trên ngữ cảnh dài."*

### Câu 8: Điểm khác biệt lớn nhất giữa mô hình của em và kiến trúc Mamba gốc của Tri Dao là gì?
* **Cách trả lời chuẩn xác (Bật Backup Slide 21):**
  > *"Thưa thầy cô, Mamba gốc là Pure SSM (mô hình không gian trạng thái thuần túy). Mamba thuần túy gặp hạn chế lớn ở khả năng truy hồi kết hợp đa truy vấn (MQAR) do toàn bộ thông tin bị nén vào vector trạng thái 1 chiều. Kiến trúc của em là mô hình **Lai (Hybrid)**: kết hợp song song cả Selective SSM và Micro-Head Linear Attention với ma trận tích lũy hiệp phương sai bậc 2 ($d_k \times d_k$), đồng thời có cơ chế Dynamic Elevator tự thích ứng pha trộn theo từng token."*

### Câu 9: Tại sao em không dùng BLEU đo tự động trên toàn bộ 34.3 triệu câu mà chỉ dùng tập con?
* **Cách trả lời chuẩn xác:**
  > *"Thưa thầy cô, tập 34.3 triệu câu là tập huấn luyện (Training Set). Để đánh giá khách quan điểm số BLEU khoa học, em đã trích xuất riêng một tập kiểm thử độc lập (Test Set) gồm hàng nghìn câu song ngữ đại diện cho 3 phong cách: Tiên hiệp Hán - Việt, Kiếm hiệp và Hội thoại đời sống chưa từng xuất hiện trong quá trình huấn luyện ạ (Mục 3.3.3)."*

### Câu 10: Nếu muốn đưa mô hình này vào ứng dụng thương mại thực tế, em sẽ triển khai như thế nào?
* **Cách trả lời chuẩn xác (Bật Backup Slide 25):**
  > *"Thưa thầy cô, nhờ ưu thế bộ nhớ đệm chỉ 1.54 MB, mô hình này có thể chạy trực tiếp trên các máy chủ đám mây giá rẻ hoặc thậm chí trên laptop người dùng. Em đã xây dựng sẵn bản Web Demo bằng Streamlit đo đạc thời gian thực và đóng gói API, sẵn sàng tích hợp vào các extension trình duyệt để dịch chương truyện dài hàng vạn chữ chỉ trong vài giây."*
