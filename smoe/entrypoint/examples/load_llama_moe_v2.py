"""
Load LLaMA MoE v2 model from file.
"""

import argparse

import torch.cuda
from transformers import AutoTokenizer

from smoe.models.mixtral import MixtralForCausalLM


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")

    model = MixtralForCausalLM.from_pretrained(args.model_path)

    """prepare data"""
    sentence_list = [
        "hi hi hi hi hi, hi hi hi hi hi, hi hi hi hi hi",
        "How are you? I'm fine, and you?",
        "<s> <unk> <unk> <unk> <unk> <unk> </s>",
        "I am stupid. Are you sure?",
        "The past is never dead. It is not even past.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(sentence_list, padding=True, return_tensors="pt")
    print(tokens)

    """forward test"""
    print("Forwarding inputs...")
    model.half()
    model.to(device)
    tokens.to(device)
    result = model.generate(**tokens, repetition_penalty=2.0, max_length=256)
    print(result)

    for i in range(result.shape[0]):
        print(result[i])
        decoded_text = tokenizer.decode(result[i], skip_special_tokens=True)
        print(decoded_text)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    main(args)
