import argparse
import os

import argparse
from common.parser import get_thresh_tasks_args
from common.logger import WandbLogger
from thresh_attn import thresh_config

import torch
from transformers import AutoModelForCausalLM, AttentionInterface
from yalis_LM import YalisLM
# from thresh_attn import thresh_attn_decode

import lm_eval
import torch._dynamo

import os

# Required to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face OpenLLM Tasks and associated metrics from April 2025
# https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard
TASKS = [
    "leaderboard_ifeval",
    # "leaderboard_bbh",
    "leaderboard_math_hard",
    # "leaderboard_gpqa",
    # "leaderboard_musr",
    # "leaderboard_mmlu_pro"
]

if __name__ == "__main__":
    # Create parser and get cli args
    parser = argparse.ArgumentParser()
    parser = get_thresh_tasks_args(parser)
    args = parser.parse_args()

    # Set thresh args
    # TODO: Propogate thresh args into thresh implementation
    thresh_config.set_thresh_tasks_args(
        args.percentile,
        args.warmup,
        args.no_txt,
        args.no_decode
    )

    # Create logger (wandb)
    if args.use_wandb:
        logger = WandbLogger(args=args, groupid="tasks")

    # run w/ yalis backend
    if args.use_thresh:
        results = lm_eval.simple_evaluate(
            model="yalis",
            model_args={
                "pretrained": args.model_id,
                "device": "cuda",
                "dtype": "fp16",
                "batch_size": 8,
                "max_length": args.sequence_length,
            },
            tasks=TASKS,
            log_samples=False,
            use_thresh=True,
            limit=0.1
        )

    else:
        results = lm_eval.simple_evaluate(
            model="yalis",
            model_args={
                "pretrained": args.model_id,
                "device": "cuda",
                "dtype": "fp16",
                "batch_size": 8,
                "max_length": args.sequence_length,
            },
            tasks=TASKS,
            log_samples=False,
            limit=0.1
        )

    print(results["results"])

    if args.use_wandb:
        logger.log_tasks(results["results"])
        logger.finish()


