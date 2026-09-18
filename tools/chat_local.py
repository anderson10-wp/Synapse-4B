import argparse
from pathlib import Path

import torch
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a local Synapse-4B model.")
    parser.add_argument(
        "--model-dir",
        default="./models/Synapse-4B-HF",
        help="Directory containing the converted Hugging Face model and tokenizer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens generated per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    return parser.parse_args()


def load_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(Fore.CYAN + f"Loading Synapse-4B from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def build_prompt(user_input: str) -> str:
    return (
        "<sistema>\n"
        "Você é a Synapse, uma inteligência artificial avançada e prestativa.\n"
        "Pense em <pensamento> antes de responder.\n"
        "</sistema>\n"
        f"<usuario> {user_input} <synapse> "
    )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    try:
        model, tokenizer = load_model(model_dir)
    except Exception as exc:
        print(Fore.RED + f"Failed to load model: {exc}")
        raise SystemExit(1)

    print(Fore.GREEN + "\nSynapse-4B loaded successfully.")
    print(Fore.GREEN + "Type 'sair', 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input(Fore.YELLOW + "Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"sair", "exit", "quit"}:
            break

        if not user_input:
            continue

        prompt = build_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt")

        print(Fore.MAGENTA + "Synapse: ", end="", flush=True)

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as exc:
            print(Fore.RED + f"Generation error: {exc}")
            continue

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=False,
        )
        print(Style.RESET_ALL + response)
        print("-" * 60)


if __name__ == "__main__":
    main()
