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
import time

app = Flask(__name__)
# Keep structure for output
app.config["JSON_SORT_KEYS"] = False

# Global configs
enable_profiling = False
tokens_to_gen = 512
system_prompt = "You are a helpful chatbot. Answer the following question.\n"
if enable_profiling:
    profiler_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
    )
else:
    profiler_context = nullcontext()

# Global caches
global_tokenizers = {}
global_engines = {}

@app.route("/v1/completions", methods=["POST"])
def infer_endpoint():
    data = request.json
    print("==> Request received: ")
    print(data)

    # Required params for completions
    # Expect a single prompt for now
    if "prompt" not in data or "model" not in data:
        return jsonify({"error": "No prompt or model provided"}), 400
    
    user_prompt = data["prompt"]
    model_id = data["model"]

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("==> Conversation created")

    # Check cache
    if model_id not in global_tokenizers:
        # Tokenizer for encoding the prompt
        global_tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    # Get value from cache
    tokenizer = global_tokenizers[model_id]

    formatted_prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # Check cache
    if model_id not in global_engines:
        # Right now set for only 1 batch size (1 prompt)
        model_config = ModelConfig(model_name=model_id, precision="bf16")
        inference_config = InferenceConfig(
            batch_size=1,
            max_length_of_generated_sequences=1024,
            top_p=0.80,
            temperature=1.0,
            tp_dims=(1,1,1)
        )
        global_engines[model_id] = LLMEngine(model_config=model_config, inference_config=inference_config)
    # Get value from cache
    engine = global_engines[model_id]
    
    with profiler_context as prof:
        output_tokens = engine.generate(
            [formatted_prompt], report_throughput=True, tokens_to_generate=tokens_to_gen
        )
        if enable_profiling:
            prof.step()
    
    output_tokens = output_tokens.cpu()
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    print("Detokenized text done, json being sent back to client :)")


    response = {
        "id": "TEMP",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "text": detokenized_text[0],
                "index": 0,
                "logprobs": None,
                "finish_reason": "TEMP"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer(user_prompt)["input_ids"]),
            "completion_tokens": len(tokenizer(detokenized_text[0])["input_ids"]),
            "total_tokens": len(tokenizer(user_prompt)["input_ids"]) + len(tokenizer(detokenized_text[0])["input_ids"])
        }
    }
    
    return jsonify(response)

if __name__ == "__main__":
    # Ideally add handling for rank 0 spinning up the server
    print("Starting Flask server on rank 0. Listening on port 5000...")
    app.run(host="0.0.0.0", port=5000)

