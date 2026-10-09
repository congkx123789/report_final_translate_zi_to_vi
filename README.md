# 🚀 My Alida Model: Neural Machine Translation (Chinese -> Vietnamese)
> **Mô hình Dịch Thuật Trung - Việt Kiến Trúc Evolving Micro-Head SSM**

---

## 📌 GIỚI THIỆU DỰ ÁN
Dự án phát triển mô hình Neural Machine Translation (NMT) thế hệ mới chuyên dụng cho cặp ngôn ngữ **Trung - Việt (ZH -> VI)**. Mô hình sử dụng kiến trúc kết hợp **Evolving Micro-Head State Space Model (SSM)** giúp dịch thuật nhanh, tiết kiệm VRAM và bảo toàn chính xác tên riêng, từ vựng Hán-Việt nhờ cơ chế **Glossary Prompt Injection**.

---

## ⚡ CÁC TÍNH NĂNG NỔI BẬT

1. **Chuẩn Hóa Ngữ Cảnh Dài (2048 Max Token & Batch 16):**
   * Hỗ trợ độ dài câu/đoạn văn lên tới **2048 Tokens**.
   * Cấu hình **Batch Size = 16** (kết hợp Gradient Accumulation = 2) tối ưu mượt mà cho GPU NVIDIA.

2. **Định Dạng Glossary Dính Sát Hán-Việt:**
   * Cấu trúc Prompt siêu gọn: `<glossary> 调整điều chỉnh 位置vị trí </glossary> <src> ... </src>`.
   * Cắt giảm **42.7% token rác**, tiết kiệm hơn 185 triệu tokens trên toàn bộ tập dữ liệu 5.5 triệu câu.

3. **Bảo Toàn 100% Dấu Xuống Dòng (`\n`):**
   * Ánh xạ trực tiếp `\n` sang **Token ID `14` (`<0x0A>`)**.
   * Chuẩn hóa tự động ngắt dòng `\r\n` (Windows) và `\r` (Mac) về `\n` (Linux) giúp đầu ra dịch giữ nguyên cấu trúc đoạn văn bản.

4. **Tập Từ Vựng Dùng Chung (24,004 Tokens Joint Vocab):**
   * Mã hóa đơn Token cho 4 thẻ điều khiển: `<glossary>`, `</glossary>`, `<src>`, `</src>`.

---

## 📁 CẤU TRÚC DỰ ÁN

```text
My_alida_model/
├── config.py                               # Cấu hình siêu tham số (2048 tokens, batch 16, BF16)
├── clean_input.py                          # Bộ chuẩn hóa 10 quy tắc & Đóng gói build_prompt()
├── vocab_tokenizer.py                      # Tokenizer 24k Joint Vocab (xử lý \n ID 14 & Control Tags)
├── dataset_loader.py                       # DataLoader (ParallelTextDataset & StreamingParallelDataset)
├── train.py                                # Script huấn luyện & Fine-tuning
├── translate.py                            # CLI & API dịch thuật (Greedy / Beam Search)
├── quick_reformat_glossary_tagged.py       # Script cập nhật tập dữ liệu Fine-tune
├── quick_reformat_glossary_tagged_output_dir.py # Script đa luồng cập nhật 235 tệp Pre-train Streaming
├── INPUT_PIPELINE_GUIDE.md                 # Tài liệu quy trình đầu vào & Chuẩn hóa
├── MODEL_ARCHITECTURE_GUIDE.md             # Tài liệu kiến trúc Evolving Micro-Head SSM
└── README.md                               # Hướng dẫn tổng quan hệ thống
```

---

## 🚀 HƯỚNG DẪN SỬ DỤNG

### 1. Dịch Thuật Trực Tiếp (Inference CLI)

* **Dịch một câu đơn:**
  ```bash
  python3 translate.py --text "他必须得抓紧时间了." --beam 3
  ```

* **Chế độ dịch tương tác terminal (Interactive Mode):**
  ```bash
  python3 translate.py --interactive --beam 3
  ```

* **Sử dụng trong Python Code:**
  ```python
  from translate import Translator

  translator = Translator()
  
  # Dịch kèm từ điển Hán-Việt (Glossary)
  result = translator.translate(
      zh_text="他必须得抓紧时间了.",
      glossary_pairs=[("抓紧", "nắm chắc"), ("时间", "thời gian")]
  )
  print("Kết quả:", result)
  ```

---

### 2. Huấn Luyện & Fine-Tuning Model

* **Chạy Fine-Tuning trên Bộ Dữ Liệu Tinh Lọc:**
  ```bash
  python3 train.py --finetune --epochs 5 --batch-size 16 --accum-steps 2
  ```

* **Chạy Pre-Training Streaming từ 235 tệp `.jsonl.gz`:**
  ```bash
  python3 train.py --streaming --epochs 5 --batch-size 16
  ```

---

## 📚 TÀI LIỆU THAM KHẢO CHI TIẾT
* 📘 [Hướng dẫn Quy trình Đầu vào (INPUT_PIPELINE_GUIDE.md)](file:///home/alida/Documents/IT/all_model/My_alida_model/INPUT_PIPELINE_GUIDE.md)
* 🧠 [Tài liệu Kiến trúc Mô hình (MODEL_ARCHITECTURE_GUIDE.md)](file:///home/alida/Documents/IT/all_model/My_alida_model/MODEL_ARCHITECTURE_GUIDE.md)
