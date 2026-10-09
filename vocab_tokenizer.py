import json
import os
import re
from typing import List, Dict
import sentencepiece as spm
from config import config

class VocabTokenizer:
    """
    Tokenizer and Vocabulary Manager powered by SentencePiece BPE (source.spm & target.spm)
    and single joint vocabulary (joint_vocab.json - 24,004 tokens).
    Provides ultra-fast subword BPE tokenization for Chinese and Vietnamese,
    with 100% single-token support for special control tags (<glossary>, </glossary>, <src>, </src>)
    and 100% loss-less newline (\\n) byte-fallback preservation (ID 14 = <0x0A>).
    """
    def __init__(
        self,
        spm_src_path: str = "vocab/source.spm",
        spm_tgt_path: str = "vocab/target.spm",
        joint_json_path: str = config.joint_vocab_path
    ):
        self.pad_idx = config.pad_idx
        self.mask_idx = config.mask_idx
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.unk_idx = config.unk_idx
        self.newline_idx = 14  # <0x0A> ASCII linefeed byte fallback

        # Special control tokens for glossary & source structure
        self.special_control_map = {
            "<glossary>": 24000,
            "</glossary>": 24001,
            "<src>": 24002,
            "</src>": 24003
        }
        self.reverse_control_map = {v: k for k, v in self.special_control_map.items()}

        # Special indices set for clean decoding
        self.special_indices = {self.pad_idx, self.mask_idx, self.bos_idx, self.eos_idx}

        # Initialize SentencePiece Processors
        if os.path.exists(spm_src_path) and os.path.exists(spm_tgt_path):
            self.sp_src = spm.SentencePieceProcessor()
            self.sp_src.load(spm_src_path)

            self.sp_tgt = spm.SentencePieceProcessor()
            self.sp_tgt.load(spm_tgt_path)

            self.use_spm = True
        else:
            self.use_spm = False

        if os.path.exists(joint_json_path):
            with open(joint_json_path, "r", encoding="utf-8") as f:
                self.vocab2idx = json.load(f).get("vocab2idx", {})
        elif self.use_spm:
            self.vocab2idx = {self.sp_src.id_to_piece(i): i for i in range(self.sp_src.get_piece_size())}
        else:
            self.vocab2idx = {}

        # Merge special control map into vocab2idx
        for token, idx in self.special_control_map.items():
            self.vocab2idx[token] = idx

        self.zh2idx = self.vocab2idx
        self.vi2idx = self.vocab2idx
        self.idx2vocab = {v: k for k, v in self.vocab2idx.items()}
        self.idx2zh = self.idx2vocab
        self.idx2vi = self.idx2vocab

        # Split pattern including special control tags and newlines
        self.special_split_pattern = re.compile(
            r'(\n+|<glossary>|</glossary>|<src>|</src>)'
        )

    @property
    def zh_vocab_size(self) -> int:
        return len(self.vocab2idx)

    @property
    def vi_vocab_size(self) -> int:
        return len(self.vocab2idx)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab2idx)

    def tokenize_zh(self, text: str) -> List[str]:
        """Tokenize Chinese text into SentencePiece BPE tokens, preserving control tags and newlines."""
        if not self.use_spm:
            return list(text)

        parts = self.special_split_pattern.split(text)
        tokens = []
        for part in parts:
            if not part:
                continue
            if part in self.special_control_map:
                tokens.append(part)
            elif part == "\n":
                tokens.append("<0x0A>")
            elif part.startswith("\n"):
                tokens.extend(["<0x0A>"] * len(part))
            else:
                tokens.extend(self.sp_src.encode(part, out_type=str))
        return tokens

    def tokenize_vi(self, text: str) -> List[str]:
        """Tokenize Vietnamese text into SentencePiece BPE tokens."""
        if not self.use_spm:
            return re.findall(r'\w+|[^\w\s]', text)

        parts = self.special_split_pattern.split(text)
        tokens = []
        for part in parts:
            if not part:
                continue
            if part == "\n":
                tokens.append("<0x0A>")
            elif part.startswith("\n"):
                tokens.extend(["<0x0A>"] * len(part))
            else:
                tokens.extend(self.sp_tgt.encode(part, out_type=str))
        return tokens

    def encode_zh(self, text: str, add_bos_eos: bool = False) -> List[int]:
        """Encode Chinese text into token IDs using BPE with single-token special tags and newline preservation."""
        ids = []
        if self.use_spm:
            parts = self.special_split_pattern.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_control_map:
                    ids.append(self.special_control_map[part])
                elif part == "\n":
                    ids.append(self.newline_idx)
                elif part.startswith("\n"):
                    ids.extend([self.newline_idx] * len(part))
                else:
                    ids.extend(self.sp_src.encode(part, out_type=int))
        else:
            ids = [self.vocab2idx.get(c, self.unk_idx) for c in text if c in self.vocab2idx]

        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def encode_vi(self, text: str, add_bos_eos: bool = True) -> List[int]:
        """Encode Vietnamese text into token IDs using BPE with newline preservation."""
        ids = []
        if self.use_spm:
            parts = self.special_split_pattern.split(text)
            for part in parts:
                if not part:
                    continue
                if part == "\n":
                    ids.append(self.newline_idx)
                elif part.startswith("\n"):
                    ids.extend([self.newline_idx] * len(part))
                else:
                    ids.extend(self.sp_tgt.encode(part, out_type=int))
        else:
            words = self.tokenize_vi(text)
            ids = [self.vocab2idx.get(w, self.unk_idx) for w in words if w in self.vocab2idx]

        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode_zh(self, token_ids: List[int], remove_specials: bool = True) -> str:
        """Decode Chinese token IDs back to text with perfect newline restoration."""
        clean_ids = []
        for idx in token_ids:
            if remove_specials and idx in self.special_indices:
                continue
            clean_ids.append(idx)

        if not self.use_spm:
            return "".join([self.idx2vocab.get(i, "") for i in clean_ids])

        result_parts = []
        current_spm_ids = []

        for idx in clean_ids:
            if idx in self.reverse_control_map:
                if current_spm_ids:
                    result_parts.append(self.sp_src.decode(current_spm_ids))
                    current_spm_ids = []
                result_parts.append(self.reverse_control_map[idx])
            elif idx == self.newline_idx:
                if current_spm_ids:
                    result_parts.append(self.sp_src.decode(current_spm_ids))
                    current_spm_ids = []
                result_parts.append("\n")
            else:
                current_spm_ids.append(idx)

        if current_spm_ids:
            result_parts.append(self.sp_src.decode(current_spm_ids))

        return "".join(result_parts)

    def decode_vi(self, token_ids: List[int], remove_specials: bool = True) -> str:
        """Decode Vietnamese token IDs back to natural text with perfect newline restoration."""
        clean_ids = [idx for idx in token_ids if not (remove_specials and idx in self.special_indices)]
        if not self.use_spm:
            return " ".join([self.idx2vocab.get(i, "") for i in clean_ids])

        result_parts = []
        current_spm_ids = []

        for idx in clean_ids:
            if idx == self.newline_idx:
                if current_spm_ids:
                    result_parts.append(self.sp_tgt.decode(current_spm_ids))
                    current_spm_ids = []
                result_parts.append("\n")
            elif idx < self.sp_tgt.get_piece_size():
                current_spm_ids.append(idx)

        if current_spm_ids:
            result_parts.append(self.sp_tgt.decode(current_spm_ids))

        return "".join(result_parts)

if __name__ == "__main__":
    tokenizer = VocabTokenizer()
    print(f"Loaded Joint Vocab Size: {tokenizer.vocab_size:,}")

    test_prompt = "<glossary> 临江Lâm Giang 确认xác nhận </glossary> <src> 他必须得抓紧时间了.\n凌伊山掏出手机,查询起了临江 </src>"
    test_vi = "Hắn phải nhanh chóng lên đường.\nLăng Y Sơn lấy điện thoại ra, tra vé máy bay."
    
    encoded_zh = tokenizer.encode_zh(test_prompt)
    decoded_zh = tokenizer.decode_zh(encoded_zh)

    encoded_vi = tokenizer.encode_vi(test_vi, add_bos_eos=False)
    decoded_vi = tokenizer.decode_vi(encoded_vi)

    print("\n[TEST PROMPT ENCODING ZH]")
    print("Raw Input Text:", repr(test_prompt))
    print("Encoded IDs   :", encoded_zh)
    print("Decoded Text  :", repr(decoded_zh))
    print("ZH Match      :", test_prompt == decoded_zh)

    print("\n[TEST TARGET ENCODING VI]")
    print("Raw Input Text:", repr(test_vi))
    print("Encoded IDs   :", encoded_vi)
    print("Decoded Text  :", repr(decoded_vi))
    print("VI Match      :", test_vi == decoded_vi)
