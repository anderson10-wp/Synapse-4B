import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw Synapse checkpoint to Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/synapse_4b_latest.pt",
        help="Path to the raw Synapse .pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="./models/Synapse-4B-HF",
        help="Destination directory for the converted model.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer.json/tokenizer.model to copy to the output directory.",
    )
    return parser.parse_args()


def convert_checkpoint(checkpoint_path: Path, output_dir: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]

    hf_dict = {
        "model.embed_tokens.weight": state_dict["token_embedding.weight"],
        "lm_head.weight": state_dict["lm_head.weight"],
        "model.norm.weight": state_dict["norm.weight"],
    }

    for layer_idx in range(config["n_layer"]):
        prefix = f"blocks.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        hf_dict[f"{hf_prefix}.input_layernorm.weight"] = state_dict[
            f"{prefix}.attn_norm.weight"
        ]
        hf_dict[f"{hf_prefix}.post_attention_layernorm.weight"] = state_dict[
            f"{prefix}.mlp_norm.weight"
        ]

        hf_dict[f"{hf_prefix}.self_attn.q_proj.weight"] = state_dict[
            f"{prefix}.attention.q_proj.weight"
        ]
        hf_dict[f"{hf_prefix}.self_attn.k_proj.weight"] = state_dict[
            f"{prefix}.attention.k_proj.weight"
        ]
        hf_dict[f"{hf_prefix}.self_attn.v_proj.weight"] = state_dict[
            f"{prefix}.attention.v_proj.weight"
        ]
        hf_dict[f"{hf_prefix}.self_attn.o_proj.weight"] = state_dict[
            f"{prefix}.attention.o_proj.weight"
        ]

        hf_dict[f"{hf_prefix}.mlp.gate_proj.weight"] = state_dict[
            f"{prefix}.mlp.gate_proj.weight"
        ]
        hf_dict[f"{hf_prefix}.mlp.up_proj.weight"] = state_dict[
            f"{prefix}.mlp.up_proj.weight"
        ]
        hf_dict[f"{hf_prefix}.mlp.down_proj.weight"] = state_dict[
            f"{prefix}.mlp.down_proj.weight"
        ]

    output_weights = output_dir / "pytorch_model.bin"
    output_config = output_dir / "config.json"

    print(f"Writing weights: {output_weights}")
    torch.save(hf_dict, output_weights)

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": config["n_embd"],
        "intermediate_size": config["intermediate_size"],
        "max_position_embeddings": config["max_seq_len"],
        "model_type": "llama",
        "num_attention_heads": config["n_head"],
        "num_hidden_layers": config["n_layer"],
        "num_key_value_heads": config["n_kv_head"],
        "rms_norm_eps": config.get("norm_eps", 1e-5),
        "rope_theta": config.get(
            "rope_theta",
            config.get("rope_base", 10000.0),
        ),
        "vocab_size": config["vocab_size"],
        "hidden_act": "silu",
        "tie_word_embeddings": True,
    }

    with output_config.open("w", encoding="utf-8") as file:
        json.dump(hf_config, file, indent=2, ensure_ascii=False)

    print(f"Writing config: {output_config}")


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        convert_checkpoint(checkpoint_path, output_dir)

        if args.tokenizer:
            tokenizer_path = Path(args.tokenizer).expanduser().resolve()
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

            destination = output_dir / tokenizer_path.name
            shutil.copy2(tokenizer_path, destination)
            print(f"Copied tokenizer: {destination}")

        print("Conversion completed successfully.")
    except KeyError as exc:
        raise SystemExit(f"Checkpoint is missing expected key: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Conversion failed: {exc}") from exc


if __name__ == "__main__":
    main()
