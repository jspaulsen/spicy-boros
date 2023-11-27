from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM


def download_model(model_path, model_name):
    path: Path = Path(model_path) / Path(model_name)

    if not path.exists():
        path.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(path, max_shard_size='200MB')
    tokenizer.save_pretrained(path, max_shard_size='200MB')


def main():
    args = ArgumentParser()
    args.add_argument('--model', type=str, required=True)
    args.add_argument('--model_path', type=str, required=False, default='models/')

    args = args.parse_args()
    download_model(args.model_path, args.model)


if __name__ == '__main__':
    main()
