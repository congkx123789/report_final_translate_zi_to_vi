import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Model Architecture Specs (Joint 24k MarianMT Architecture)
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 8
    num_decoder_layers: int = 2
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048  # Nâng cấp Max Sequence Length lên 2048 Tokens cho cả Input & Output

    # Special Token Indices (Aligned with SentencePiece BPE 24k Joint Vocab)
    unk_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    pad_idx: int = 3
    mask_idx: int = 3

    # Special Token Strings
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"

    # Hardware & Performance Specs (Optimized for sm_120 Blackwell Architecture)
    use_bf16: bool = True
    use_flash_attn: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # File Paths
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    joint_vocab_path: str = os.path.join(project_root, "vocab", "joint_vocab.json")
    zh_vocab_path: str = joint_vocab_path
    vi_vocab_path: str = joint_vocab_path
    dataset_dir: str = os.path.join(project_root, "dataset", "master_mixed_train_dataset")
    checkpoint_dir: str = os.path.join(project_root, "checkpoints")
    checkpoint_path: str = os.path.join(project_root, "checkpoints", "ssm_model_latest.pt")

    # Training Specs
    batch_size: int = 16  # Đã giảm Batch Size xuống 16 theo đúng yêu cầu
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    clip_grad: float = 1.0

config = Config()
