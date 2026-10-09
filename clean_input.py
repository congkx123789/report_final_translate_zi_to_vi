"""
clean_input.py: Module tiền xử lý và làm sạch văn bản tiếng Trung, chuẩn hóa prompt dịch.
"""

import re

def clean_input(text: str) -> str:
    """Làm sạch các ký tự rác, chuẩn hóa khoảng trắng nhưng GIỮ NGUYÊN dấu xuống dòng \n giữa các đoạn."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def build_prompt(src_text: str, glossary: dict = None) -> str:
    """
    Tạo prompt chuẩn cho mô hình:
    - Có glossary: <glossary> key1 val1 key2 val2 </glossary> <src> text </src>
    - Không glossary: <src> text </src>
    """
    clean_src = clean_input(src_text)
    
    if glossary and len(glossary) > 0:
        glossary_parts = []
        for k, v in glossary.items():
            glossary_parts.append(f"{k}{v}")
        glossary_str = " ".join(glossary_parts)
        return f"<glossary> {glossary_str} </glossary> <src> {clean_src} </src>"
    
    return f"<src> {clean_src} </src>"
