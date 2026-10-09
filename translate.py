import argparse
import os
import torch
import torch.nn.functional as F
from config import config
from vocab_tokenizer import VocabTokenizer
from models.micro_head_ssm import EvolvingHeadSSMModel, load_checkpoint_compatible
from clean_input import clean_input, build_prompt

class SSMTranslator:
    """
    Inference Pipeline for Evolving Micro-Head SSM Chinese (ZH) to Vietnamese (VI) Translation.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = VocabTokenizer()

        self.model = EvolvingHeadSSMModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.d_model,
            num_micro_heads=config.nhead,
            num_layers=config.num_encoder_layers,
            window_sizes=(5, 20, None),
            dim_feedforward=config.dim_feedforward,
            deep_ssm_expand_factor=2.0,
            dropout=0.0,
            max_seq_len=config.max_seq_len,
            pad_idx=config.pad_idx,
        ).to(self.device)

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            candidates = [
                "checkpoints/ssm_model_latest.pt",
                "checkpoints/ssm_model_step_685000.pt",
                "checkpoints/ssm_model_best.pt",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    checkpoint_path = cand
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📦 Loading Checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_checkpoint_compatible(self.model, state_dict)

            step = checkpoint.get("global_step", "N/A")
            loss = checkpoint.get("best_loss", "N/A")
            print(f"✓ Checkpoint Loaded Successfully! (Global Step: {step} | Best Loss: {loss})")
        else:
            print(f"⚠️ Warning: Checkpoint '{checkpoint_path}' not found! Model initialized with random weights.")

        self.model.eval()

    @torch.no_grad()
    def translate(
        self,
        zh_text: str,
        glossary_pairs: list = None,
        max_len: int = 2048,
        beam_size: int = 3,
        suppress_unk: bool = True
    ) -> str:
        """
        Translates input Chinese sentence into Vietnamese using Greedy Search or Beam Search.
        """
        # Formulate standard training prompt (with or without glossary)
        if not zh_text.startswith("<src>") and not zh_text.startswith("<glossary>"):
            prompt_str = build_prompt(zh_text, glossary_pairs)
        else:
            prompt_str = clean_input(zh_text)

        src_ids = self.tokenizer.encode_zh(prompt_str, add_bos_eos=False)
        if not src_ids:
            return ""

        prompt = src_ids + [config.eos_idx, config.bos_idx]
        tokens_tensor = torch.tensor([prompt], dtype=torch.long, device=self.device)

        if beam_size <= 1:
            generated_ids = []
            full_seq = list(prompt)
            for _ in range(max_len):
                seq_tensor = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                logits, _ = self.model(seq_tensor)
                next_logits = logits[0, -1]
                del logits

                if suppress_unk:
                    next_logits[config.unk_idx] = float("-inf")

                # Ngăn chặn lặp lại chuỗi 3 token liên tiếp (no-repeat 3-gram)
                if len(generated_ids) >= 3:
                    last_trigram = tuple(generated_ids[-2:])
                    for i in range(len(generated_ids) - 2):
                        if (generated_ids[i], generated_ids[i+1]) == last_trigram:
                            banned_tok = generated_ids[i+2]
                            next_logits[banned_tok] = float("-inf")

                next_token = torch.argmax(next_logits, dim=-1).item()

                if next_token == config.eos_idx or next_token == config.pad_idx:
                    break

                generated_ids.append(next_token)
                full_seq.append(next_token)
            return self.tokenizer.decode_vi(generated_ids)
        else:
            beams = [(0.0, prompt, [])]
            completed_beams = []

            for _ in range(max_len):
                new_beams = []
                for score, full_seq, gen_seq in beams:
                    if gen_seq and gen_seq[-1] == config.eos_idx:
                        completed_beams.append((score, gen_seq))
                        continue

                    seq_tensor = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                    logits, _ = self.model(seq_tensor)
                    next_logits = logits[0, -1]
                    del logits

                    if suppress_unk:
                        next_logits[config.unk_idx] = float("-inf")

                    log_probs = F.log_softmax(next_logits, dim=-1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                    for k in range(beam_size):
                        tok_id = topk_ids[k].item()
                        tok_score = topk_log_probs[k].item()
                        new_beams.append((score + tok_score, full_seq + [tok_id], gen_seq + [tok_id]))

                if not new_beams:
                    break

                beams = sorted(new_beams, key=lambda x: x[0] / max(len(x[2]), 1), reverse=True)[:beam_size]

            completed_beams.extend([(score, gen_seq) for score, _, gen_seq in beams])
            completed_beams = sorted(completed_beams, key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            best_gen_ids = completed_beams[0][1]
            return self.tokenizer.decode_vi(best_gen_ids)

    def translate_sentence(self, zh_text: str, glossary_pairs: list = None, max_len: int = 256, beam_size: int = 3) -> str:
        return self.translate(zh_text, glossary_pairs=glossary_pairs, max_len=max_len, beam_size=beam_size)

Translator = SSMTranslator

def main():
    parser = argparse.ArgumentParser(description="ZH -> VI Translation CLI for Evolving Head SSM Model")
    parser.add_argument("--text", type=str, default="我们需要继续努力", help="Chinese sentence to translate")
    parser.add_argument("--beam", type=int, default=3, help="Beam size for decoding (default: 3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal translation mode")
    args = parser.parse_args()

    translator = Translator(checkpoint_path=args.checkpoint, device=args.device)

    if args.interactive:
        print("\n=== Interactive Evolving-Head SSM ZH -> VI Translator (Type 'exit' to quit) ===")
        while True:
            try:
                user_input = input("\n[ZH Input]: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                translation = translator.translate(user_input, beam_size=args.beam)
                print(f"[VI Output]: {translation}")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print(f"\n[ZH Input] : {args.text}")
        translation = translator.translate(args.text, beam_size=args.beam)
        print(f"[VI Output]: {translation}")

if __name__ == "__main__":
    main()
