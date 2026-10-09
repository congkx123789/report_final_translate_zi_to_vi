# 📚 HƯỚNG DẪN CHI TIẾT QUY TRÌNH ĐẦU VÀO (INPUT PIPELINE GUIDE)
> **Dự án Model Dịch Thuật Trung - Việt (SSM Architecture)**  
> *Cập nhật chuẩn hóa: Chuẩn 2048 Tokens, Batch 16, Bảo toàn 100% Dấu xuống dòng (`\n`), Tối ưu Glossary Dính Sát Hán-Việt.*

---

## 📑 MỤC LỤC
1. [Các Cách Thức Input Vào Model (Input Modalities)](#1-các-cách-thức-input-vào-model)
2. [Chi Tiết Quy Tắc Chuẩn Hóa Dữ Liệu (`clean_input.py`)](#2-chi-tiết-quy-tắc-chuẩn-hóa-dữ-liệu)
3. [Xử Lý Ký Tự Xuống Dòng `\n` & Các Hệ Điều Hành](#3-xử-lý-ký-tự-xuống-dòng-n--các-hệ-điều-hành)
4. [Cơ Chế Tokenizer & Mã Hóa Special Control Tokens](#4-cơ-chế-tokenizer--mã-hóa-special-control-tokens)
5. [Ví Dụ Tích Hợp Code Thực Tế (Inference & Training)](#5-ví-dụ-tích-hợp-code-thực-tế)

---

## 1. CÁC CÁCH THỨC INPUT VÀO MODEL

Model hỗ trợ **2 định dạng Prompt đầu vào chính** và tự động xử lý mượt mà thông qua hàm `build_prompt(zh_text, glossary_pairs)`:

### 🔹 CÁCH 1: Prompt Có Glossary (Có Từ Điển Hán-Việt Hỗ Trợ)
* **Bối cảnh sử dụng:** Khi người dùng cung cấp danh sách từ vựng/tên riêng (Glossary) hoặc hệ thống tự động trích xuất từ điển.
* **Định dạng Prompt:**
  ```text
  <glossary> 调整điều chỉnh 位置vị trí 准备chuẩn bị </glossary> <src> 然后他慢悠悠地捡起狙击步枪,调整着枪口位置,准备掩护李再安突围. </src>
  ```
* **Đặc điểm:**
  * Từ Hán và Việt dính liền nhau (không dấu hai chấm `:` hay khoảng trắng thừa).
  * Giới hạn tối đa **6 cặp từ quan trọng nhất / câu** (để giữ cho Prompt gọn nhẹ ~37-45 tokens).
  * Tiết kiệm **42.7% token rác** so với định dạng cũ.

### 🔹 CÁCH 2: Prompt Plain (Không Có Glossary / Dịch Thẳng Văn Bản Thô)
* **Bối cảnh sử dụng:** Dịch các câu thông thường, văn bản không có thuật ngữ chuyên ngành hoặc người dùng không cung cấp từ điển.
* **Định dạng Prompt:**
  ```text
  <src> 然后他慢悠悠地捡起狙击步枪,调整着枪口位置,准备掩护李再安突围. </src>
  ```
* **Đặc điểm:**
  * Không xuất hiện thẻ `<glossary>`.
  * Model vẫn hiểu và tiến hành dịch thuật chính xác dựa trên thẻ ngữ cảnh `<src> ... </src>`.

---

## 2. CHI TIẾT QUY TẮC CHUẨN HÓA DỮ LIỆU

Tất cả đầu vào (dù là văn bản thô hay chuỗi đã gán thẻ) đều được xử lý qua `clean_input(text)` theo 10 quy tắc nghiêm ngặt:

| STT | Quy Tắc Chuẩn Hóa | Nội Dung Chi Tiết & Ví Dụ |
| :---: | :--- | :--- |
| **1** | **Bảo vệ Thẻ Control** | Không phá vỡ hay bóp méo 4 thẻ: `<glossary>`, `</glossary>`, `<src>`, `</src>`. |
| **2** | **Ngoặc Kép CJK & Ngoặc Sách** | Chuyển 《》 “ ” 「 」 『 』 « » $\rightarrow$ `"` (Nháy kép chuẩn). |
| **3** | **Dấu Câu Fullwidth CJK** | Chuyển ， 。 ！ ？ ： ； （ ） ［ ］ $\rightarrow$ `, . ! ? : ; ( ) [ ]`. |
| **4** | **Chữ & Số Fullwidth (NFKC)** | Chuyển số １２３ $\rightarrow$ `123`, ① $\rightarrow$ `1`, ℃ $\rightarrow$ `°C`. |
| **5** | **Gạch Ngang & Sóng** | Chuyển – — ― − ～ 〜 $\rightarrow$ `-`. |
| **6** | **Dấu Chấm Lửng** | Chuyển ký tự 1 ô … $\rightarrow$ `...` (3 dấu chấm). |
| **7** | **Bảo Toàn Xuống Dòng** | Giữ nguyên dấu ngắt đoạn `\n`, không ép dấu xuống dòng thành khoảng trắng. |
| **8** | **Lọc Ký Tự Ẩn / PUA** | Xóa sạch ký tự vô hình (`\u200b`, `\ufeff`), BOM, Font rác PUA (`0xE000-0xF8FF`). |
| **9** | **Ký Tự Biểu Tượng** | Chuyển ★ ● ◆ $\rightarrow$ `*`, ✓ ✔ $\rightarrow$ `v`, ✕ ✖ $\rightarrow$ `x`. |
| **10** | **Khoảng Trắng Ngang** | Thu gọn các dấu cách `space` / `tab` liên tiếp thành 1 dấu cách duy nhất. |

---

## 3. XỬ LÝ KÝ TỰ XUỐNG DÒNG `\n` & CÁC HỆ ĐIỀU HÀNH

### ❓ Có sợ lệch định dạng giữa các hệ điều hành (Windows / Mac / Linux) không?
👉 **KHÔNG! HỆ THỐNG ĐÃ ĐƯỢC CHUẨN HÓA NGUYÊN BẢN:**

1. **Chuẩn Hóa Chuỗi Trong `clean_input.py`:**
   ```python
   # Đưa toàn bộ các kiểu ngắt dòng về LF \n duy nhất của Linux
   text = text.replace('\r\n', '\n').replace('\r', '\n')
   ```
   * Windows (`\r\n`) $\rightarrow$ Chuyển thành `\n`.
   * Mac cũ (`\r`) $\rightarrow$ Chuyển thành `\n`.
   * Linux (`\n`) $\rightarrow$ Giữ nguyên `\n`.

2. **Mã Hóa Tokenizer (`VocabTokenizer`):**
   * Ký tự `\n` được ánh xạ cố định trực tiếp vào **Token ID `14` (`<0x0A>`)**.
   * Khi Model đọc: `\n` $\rightarrow$ ID `14`.
   * Khi Model sinh ra văn bản dịch: ID `14` $\rightarrow$ sinh ra đúng ký tự `\n` ngắt dòng chuẩn.
   * Kết quả thử nghiệm: **100% Khớp nguyên bản (Match Exact = True)**, không bị lỗi `<unk>` (ID 0).

---

## 4. CƠ CHẾ TOKENIZER & MÃ HÓA SPECIAL CONTROL TOKENS

Bảng từ điển `joint_vocab.json` chứa **24.004 Tokens**, tích hợp sẵn các Token điều khiển đặc biệt:

| Token String | Token ID | Vai Trò Trong Hệ Thống |
| :--- | :---: | :--- |
| `<glossary>` | `24000` | Mở đầu vùng chứa các cặp từ Hán-Việt gợi ý. |
| `</glossary>` | `24001` | Kết thúc vùng từ gợi ý. |
| `<src>` | `24002` | Mở đầu vùng chứa văn bản Tiếng Trung gốc cần dịch. |
| `</src>` | `24003` | Kết thúc vùng văn bản gốc. |
| `<0x0A>` (`\n`) | `14` | Ký tự xuống dòng ngắt đoạn văn bản. |
| `<s>` (BOS) | `1` | Token bắt đầu chuỗi đích (Target Output). |
| `</s>` (EOS) | `2` | Token kết thúc chuỗi dịch. |
| `<pad>` (PAD) | `3` | Token chèn đệm chiều dài câu. |

---

## 5. VÍ DỤ TÍCH HỢP CODE THỰC TẾ

### 💡 1. Sử Dụng Trong Suy Luận / Inference Dịch Thuật:
```python
from clean_input import build_prompt, clean_input
from vocab_tokenizer import VocabTokenizer

tokenizer = VocabTokenizer()

# Trường hợp A: Đưa vào đoạn văn bản có từ điển
zh_raw = "他必须得抓紧时间了.\n凌伊山掏出手机,查询起了临江市的机票."
glossary_pairs = [("临江", "Lâm Giang"), ("机票", "vé máy bay")]

# 1. Đóng gói Prompt chuẩn
prompt = build_prompt(zh_raw, glossary_pairs)
print("Prompt Input:", prompt)

# 2. Tokenize sang IDs cho Model
input_ids = tokenizer.encode_zh(prompt)
print("Encoded Input IDs:", input_ids)

# 3. Sau khi Model sinh ra output IDs (vi_output_ids):
# vi_text = tokenizer.decode_vi(vi_output_ids)
```

### 💡 2. Cấu Hình Huấn Luyện (Training Config):
```python
# File: config.py
max_seq_len: int = 2048  # Ngữ cảnh dài 2048 Tokens
batch_size: int = 16    # 16 Samples / Batch GPU
```

---
*Tài liệu này được lưu trữ tại `INPUT_PIPELINE_GUIDE.md` để làm chuẩn tham chiếu cho toàn bộ dự án.*
