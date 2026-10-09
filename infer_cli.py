#!/usr/bin/env python3
"""
Công cụ dòng lệnh dịch thuật chuẩn dành cho mô hình SSM.
Cho phép dịch trực tiếp từ chuỗi, dịch file văn bản hoặc chạy test xuất báo cáo Markdown.
"""

import os
import sys
import argparse
import torch
from translate import Translator
from vocab_tokenizer import VocabTokenizer

def main():
    parser = argparse.ArgumentParser(description="Công cụ dịch thuật mô hình SSM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ssm_model_step_28000.pt", help="Đường dẫn file checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị chạy suy luận")
    parser.add_argument("--text", type=str, default=None, help="Chuỗi tiếng Trung hoặc prompt <src>...</src> cần dịch")
    parser.add_argument("--input-file", type=str, default=None, help="File văn bản đầu vào chứa các câu/đoạn cần dịch")
    parser.add_argument("--output-file", type=str, default=None, help="File đầu ra lưu kết quả dịch (.txt hoặc .md)")
    parser.add_argument("--beam-size", type=int, default=1, help="Kích thước Beam Search (Mặc định 1: Greedy)")
    parser.add_argument("--max-len", type=int, default=256, help="Độ dài tối đa của câu dịch đầu ra")
    
    args = parser.parse_args()
    
    # Chọn device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"📦 Đang nạp mô hình từ: {args.checkpoint} trên thiết bị [{device.upper()}]...")
    translator = Translator(checkpoint_path=args.checkpoint, device=device)
    
    # Trường hợp 1: Dịch 1 câu trực tiếp từ dòng lệnh
    if args.text:
        prompt = args.text.strip()
        if not prompt.startswith("<src>") and not prompt.startswith("<glossary>"):
            prompt = f"<src> {prompt} </src>"
            
        print("\n📥 INPUT :", prompt)
        res = translator.translate(prompt, max_len=args.max_len, beam_size=args.beam_size)
        print("🤖 OUTPUT:", res)
        
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(res + "\n")
            print(f"✓ Đã lưu kết quả ra file: {args.output_file}")
        return

    # Trường hợp 2: Dịch toàn bộ file đầu vào
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"❌ Lỗi: Không tìm thấy file {args.input_file}")
            sys.exit(1)
            
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        print(f"🚀 Bắt đầu dịch {len(lines)} đoạn từ file {args.input_file}...")
        results = []
        for idx, line in enumerate(lines, 1):
            prompt = line if line.startswith("<") else f"<src> {line} </src>"
            out = translator.translate(prompt, max_len=args.max_len, beam_size=args.beam_size)
            results.append((prompt, out))
            print(f"[{idx}/{len(lines)}] Hoàn tất.")
            
        if args.output_file:
            is_md = args.output_file.endswith(".md")
            with open(args.output_file, "w", encoding="utf-8") as f:
                if is_md:
                    f.write(f"# 📜 KẾT QUẢ DỊCH TỪ FILE `{args.input_file}`\n\n---\n\n")
                    for p, o in results:
                        f.write(f"### 📥 Nguồn:\n```text\n{p}\n```\n\n### 🤖 Bản dịch:\n```text\n{o}\n```\n\n---\n\n")
                else:
                    for _, o in results:
                        f.write(o + "\n")
            print(f"✓ Đã lưu toàn bộ bản dịch ra: {args.output_file}")
        else:
            print("\n=== KẾT QUẢ DỊCH ===")
            for _, o in results:
                print(o)
        return

    print("ℹ️ Vui lòng cung cấp --text hoặc --input-file để dịch. Sử dụng --help để xem hướng dẫn.")

if __name__ == "__main__":
    main()
