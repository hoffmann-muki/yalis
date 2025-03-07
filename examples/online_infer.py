try:
    from mpi4py import MPI
except ImportError:
    pass

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile
_KinetoProfile._get_distributed_info = lambda self: None

from contextlib import nullcontext
from flask import Flask, request, jsonify

app = Flask(__name__)

if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # profile the run or not
    enable_profiling = True

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size=1, # single batch prompts for now 
                                       max_length_of_generated_sequences=1024,
                                       top_p=0.80,
                                       temperature=1.0)

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
        )
    else:
        profiler_context = nullcontext()

    @app.route("/infer", methods=["POST"])
    def infer_endpoint():
        data = request.json
        user_prompt = data.get("prompt", "")

        with profiler_context as prof:
            output_tokens = engine.generate(
                [user_prompt],
                report_throughput=False,
                tokens_to_generate=512 # hard set to 512 for now
            )
            if enable_profiling:
                prof.step()

        output_tokens = output_tokens.cpu()

        # Decode the token IDs into text
        detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return jsonify(
                {"prompt": user_prompt,
                 "output": detokenized_text[0]})

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        print_rank0("Starting Flask server on rank 0. Listening on port 5000...")
        app.run(host="0.0.0.0", port=5000)
