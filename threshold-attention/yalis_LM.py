from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from yalis import ModelConfig, InferenceConfig, LLMEngine
from tqdm.auto import tqdm


# Heavily inspired by vLLM's implementation of lm_eval's LM implementation
# https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075bce/lm_eval/models/vllm_causallms.py#L218
@register_model("yalis")
class YalisLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 4096
    _DEFAULT_MAX_GEN    = 256

    def __init__(
        self,
        pretrained: str = "meta-llama/Meta-Llama-3-8B",
        dtype: str = "bf16",
        batch_size: int = 8,
        max_length: Optional[int] = None,
        use_thresh: bool = False,
        max_gen_toks: int = _DEFAULT_MAX_GEN,
        device: str = "cuda",
        **ignored
    ):
        super().__init__()

        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length) if max_length else self._DEFAULT_MAX_LENGTH
        self.max_gen_toks = int(max_gen_toks)
        self.use_thresh = use_thresh
        # TODO implement thresh stuff

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        model_config = ModelConfig(model_name=pretrained, precision=dtype)
        inference_config  = InferenceConfig(
            batch_size = self.batch_size,
            max_length_of_generated_sequences = self.max_length,
            tp_dims = (1,1,1),
            explicitly_use_flash_kernel = False,
        )
        self.engine = LLMEngine(model_config, inference_config )

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, s, add_special_tokens=False, **kw):
        return self.tokenizer.encode(s, add_special_tokens=add_special_tokens, **kw)

    def tok_decode(self, t):
        return self.tokenizer.decode(t, skip_special_tokens=True)

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # One shot forward pass
        B, T = tokens.shape
        pos = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            out = self.engine.model(tokens, pos)["logits"]
        return out

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str,str], List[int], List[int]]],
        disable_tqdm: bool = False
    ) -> List[Tuple[float,bool]]:
        results = []
        # chunk requests
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]
            # build [B, L] tensor
            seqs = [ctx + cont for (_,ctx,cont) in batch]
            tokens = torch.tensor(seqs, device=self.device)
            logits = self._forward(tokens)  # [B, L, V]
            logp = torch.log_softmax(logits, dim=-1)
            for j, (_, ctx, cont) in enumerate(batch):
                start = len(ctx)
                # gather log-probs of each continuation token
                lp = logp[j, start-1:-1]
                cont_ids = torch.tensor(cont, device=self.device).unsqueeze(1)
                token_ll = lp.gather(1, cont_ids).squeeze(1).sum().item()
                results.append((token_ll, True))  # greedy==True
        return results

    def loglikelihood_rolling(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False
    ) -> List[float]:
        out = []
        for (text,) in requests:
            windows = list(map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=self.tok_encode(text),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length - 1,
                    context_len=1,
                )
            ))
            windows = [(None,) + w for w in windows]
            ll = sum(x[0] for x in self._loglikelihood_tokens(windows))
            out.append(ll)
        return out


    def generate_until(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        # gen_kwargs**,
    ) -> List[str]:
        # Unpack prompts + gen-kwargs
        contexts, all_kwargs = zip(*(r.args for r in requests))
        prompts = list(contexts)
        gen_kwargs = dict(all_kwargs[0])
        gen_kwargs.pop("until", None)
        max_new = gen_kwargs.pop("max_gen_toks", self.max_gen_toks)

        outputs = []

        # Choose iterator: tqdm or plain range
        total = len(prompts)
        batch_steps = range(0, total, self.batch_size)
        if not disable_tqdm:
            batch_steps = tqdm(batch_steps, total=(total + self.batch_size - 1)//self.batch_size,
                    desc="YALIS gen", unit="batch")

        # Handle batching of prompts
        for batch_step in batch_steps:
            batch_prompts = prompts[batch_step : batch_step + self.batch_size]
            
            # Breaks due to uneven final batch size, so we manually reset the kv cache and set it to size of len(batch_prompts)
            self.engine.model.clear_kv_cache()
            self.engine.model.set_kv_cache(
                batch_size=len(batch_prompts),
                device=self.device,
                dtype=self.engine.dtype,
            )
            
            ret = self.engine.generate(
                batch_prompts,
                tokens_to_generate = max_new,
                report_throughput  = False,
            )
            
            # grab the tensor as the first element
            token_tensor = ret[0] if isinstance(ret, tuple) else ret

            # detach & decode
            batch_ids = token_tensor.cpu().tolist()
            for ids in batch_ids:
                outputs.append(self.tok_decode(ids))

        return outputs