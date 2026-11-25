# Owner: TBD (RL training)
"""GRPO + LoRA training on GSM8K that saves policy adapter checkpoints.

The saved ``checkpoint-<step>`` folders are the policies whose drafts are
studied by ``src.selfspec`` (``run.py acceptance``) and served by vLLM through
``lora_adapter`` (``run.py specdec_sample --config config/selfspec_sample.json``).

Training uses the GSM8K *train* split and the shared numeric reward; every
evaluation in this repository uses the *test* split, so they never overlap.
TRL, peft, and datasets are imported lazily.
"""

from __future__ import annotations

from typing import Any

from src.data import format_prompt, load_gsm8k
from src.reward import reward


def _completion_text(completion: Any) -> str:
    """Return text for a plain-string or conversational TRL completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    raise TypeError(f"unsupported completion type {type(completion).__name__}.")


def gsm8k_correctness(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    """TRL reward function: 1.0 for a numerically correct final answer, else 0.0.

    TRL passes dataset columns as keyword arguments, so ``answer`` holds the
    gold final answers aligned with ``completions``.
    """
    if len(completions) != len(answer):
        raise ValueError("completions and answers must align.")
    return [reward(_completion_text(c), gold) for c, gold in zip(completions, answer)]


def build_dataset(examples: list[dict], tokenizer: Any) -> Any:
    """Build a TRL prompt dataset with the same chat prompts as evaluation."""
    from datasets import Dataset

    return Dataset.from_list([
        {"prompt": format_prompt(example["question"], tokenizer), "answer": example["answer"]}
        for example in examples
    ])


def train_grpo(config: dict) -> str:
    """Train a LoRA policy with GRPO and return the output directory.

    Config keys: ``target_model``, ``n_train``, ``seed``, ``output_dir``,
    ``lora`` (peft ``LoraConfig`` kwargs), and ``grpo`` (TRL ``GRPOConfig``
    kwargs such as learning_rate, num_generations, max_completion_length,
    temperature, max_steps, save_steps). Set ``grpo.save_steps`` to the
    checkpoint spacing you want to study; each checkpoint stores the adapter.
    """
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    examples = load_gsm8k("train", config["n_train"], config["seed"])
    dataset = build_dataset(examples, tokenizer)
    args = GRPOConfig(output_dir=config["output_dir"], seed=config["seed"], **config["grpo"])
    trainer = GRPOTrainer(
        model=config["target_model"],
        reward_funcs=[gsm8k_correctness],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**config["lora"]),
    )
    trainer.train()
    trainer.save_model(f"{config['output_dir']}/final")
    print(f"GRPO adapters saved under {config['output_dir']}", flush=True)
    return config["output_dir"]
