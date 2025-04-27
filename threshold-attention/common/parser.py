# Adding parser arguments
def get_thresh_common_args(parser):
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B", help="Model ID from huggingface")
    parser.add_argument("--sequence-length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--use-axonn", action='store_true', default=False, help="Shard a model using AxoNN")
    parser.add_argument("--use-wandb", action='store_true', default=False, help="Use wandb")

    parser.add_argument("--use-thresh", action='store_true', default=False, help="Use thresh attention")
    parser.add_argument("--no-txt", action='store_true', default=False, help="Collect and save data to txt")
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=32)
    return parser

def get_thresh_ppl_args(parser):
    parser.add_argument("--dataset", type=str, default="wikitext-test", help="Dataset (wikitext-test, wikitext-valid, bookcorpus, c4)")
    parser.add_argument("--no-json", action='store_true', default=False, help="Collect and save data to json")
    parser = get_thresh_common_args(parser)
    return parser

def get_thresh_tasks_args(parser):
    parser.add_argument("--no-decode", action='store_true', default=False, help="Use the non-generative decoding version of thresh attention")
    parser = get_thresh_common_args(parser)
    return parser