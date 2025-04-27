# LAUNCH FROM OUTER YALIS FOLDER
import argparse
import subprocess
import os

# TODO: Get run_tasks.py working
EXPERIMENT_CHOICES = ["run_ppl.py", "run_tasks.py"]

MODEL_CHOICES = {
    "Llama-3.1-8B" : {"model_id": "meta-llama/Llama-3.1-8B",
                    "seq_len": 4096, 
                    "num_gpus": 1, 
                    "category": "small"},
    "Qwen2.5-7B" : {"model_id": "Qwen/Qwen2.5-7B",
                    "seq_len": 4096, 
                    "num_gpus": 1, 
                    "category": "small"},
    "Mistral-7B-v0.1" : {"model_id": "mistralai/Mistral-7B-v0.1",
                    "seq_len": 4096, 
                    "num_gpus": 1, 
                    "category": "small"},
}
TIMES = {
    "small": {
        "run_ppl.py": {
            "hf": "01:00:00",
            "thresh": "01:00:00",
        },
        "run_tasks.py": {
            "hf": "02:00:00",
            "thresh": "02:30:00",
        }
    },
    "medium": {
        "run_ppl.py": {
            "hf": "02:00:00",
            "thresh": "02:00:00",
        },
        "run_tasks.py": {
            "hf": "03:00:00",
            "thresh": "03:00:00",
        }
    },
    "large": {
        "run_ppl.py": {
            "hf": "03:00:00",
            "thresh": "03:00:00",
        },
        "run_tasks.py": {
            "hf": "04:00:00",
            "thresh": "04:00:00",
        }
    }
}

SEQUENCE_LENGTH_CHOICES = [512, 1024, 2048, 4096, 8192]

PERCENTILE_CHOICES = [0.25, 0.5, 0.75, 0.875, 0.95]
WARMUP_CHOICES = [16, 32, 64, 128, 256]

DATASET_CHOICES = ["wikitext-test", "wikitext-valid", "bookcorpus", "c4"]

METHOD_CHOICES = ["hf", "thresh"]

def make_command(experiment, method, model, percentile, warmup, dataset, use_axonn, use_wandb, sequence_length):
    model_config = MODEL_CHOICES[model]
    parts = [
        f"{experiment}",
        f"--model-id {model_config['model_id']}",
    ]
    
    if sequence_length == -1:
        parts.append(f"--sequence-length {model_config['seq_len']}")
    else:
        parts.append(f"--sequence-length {sequence_length}")

    if experiment =="run_ppl.py":
        parts.append(f"--dataset {dataset}")

    if method == "thresh":
        parts.append(f"--use-thresh")
        parts.append(f"--percentile {percentile}")
        parts.append(f"--warmup {warmup}")

    if use_wandb:
        parts.append("--use-wandb")
    if use_axonn:
        parts.append("--use-axonn")

    return " ".join(parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=False, help="Don't launch batch scripts")
    
    parser.add_argument('--experiments', type=str, nargs='+', choices=EXPERIMENT_CHOICES)
    parser.add_argument('--methods', type=str, nargs='+', choices=METHOD_CHOICES)
    
    parser.add_argument('--models', type=str, nargs='+', choices=MODEL_CHOICES.keys(), default=MODEL_CHOICES.keys())
    parser.add_argument('--sequence-length', type=int, choices=SEQUENCE_LENGTH_CHOICES, default=-1)
    
    parser.add_argument('--use-wandb', action='store_true', default=False, help="Use wandb")
    parser.add_argument('--use-axonn', action='store_true', default=False, help="Shard a model using AxoNN")
    
    parser.add_argument('--percentiles', type=float, nargs='+', default=PERCENTILE_CHOICES)
    parser.add_argument('--warmups', type=int, nargs='+', default=WARMUP_CHOICES)

    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES, default="")
    
    args = parser.parse_args()

    # load our template once
    with open("experiments/template.sh") as f:
        template = f.read()
    
    visited = set()
    for experiment in args.experiments:
        for method in args.methods:
            for model in args.models:
                for percentile in args.percentiles:
                    for warmup in args.warmups:
                        if experiment == "run_ppl.py":
                            if method == "hf":
                                job_name = f"{experiment[:-3]}_{method}_{model}_{args.dataset}"
                            elif method == "thresh":
                                job_name = f"{experiment[:-3]}_{method}_{model}_{args.dataset}_p{percentile}_w{warmup}"
                        else:
                            if method == "hf":
                                job_name = f"{experiment[:-3]}_{method}_{model}"
                            elif method == "thresh":
                                job_name = f"{experiment[:-3]}_{method}_{model}_p{percentile}_w{warmup}"

                        # Prevent duplicate jobs
                        if job_name in visited:
                            break
                        visited.add(job_name)
                        model_config = MODEL_CHOICES[model]

                        # Compute # of nodes & gpus
                        nodes = (model_config["num_gpus"] + 3) // 4
                        gpus = model_config["num_gpus"]

                        # Determine time needed
                        time = TIMES[model_config["category"]][experiment][method]

                        # Create command
                        cmd = make_command(experiment, method, model, percentile, warmup, args.dataset, args.use_axonn, args.use_wandb, args.sequence_length)
                        
                        # Metadata
                        output_file = f"experiments/logs/{job_name}.out"
                        os.makedirs("experiments/logs", exist_ok=True)

                        # Create script
                        script = template.format(
                            experiment = job_name,
                            output_file = output_file,
                            nodes = nodes,
                            gpus = gpus,
                            time = time,
                            cmd = cmd
                        )

                        # Dump to a .sh
                        script_path = f"experiments/scripts/{job_name}.sh"
                        os.makedirs("experiments/scripts", exist_ok=True)
                        with open(script_path, "w") as w:
                            w.write(script)

                        print(job_name)
                        if not args.dry_run:
                            # Run script
                            subprocess.run(["sbatch", script_path])
