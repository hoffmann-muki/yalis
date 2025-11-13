from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import torch


class RequestStatus(Enum):
    """
    Lifecycle states for a request that is being processed by the engine.
    """

    WAITING = auto()
    PREFILLING = auto()
    DECODE = auto()
    FINISHED = auto()


@dataclass
class SamplingOptions:
    """
    Per-request sampling configuration.
    """

    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 0.80
    ignore_eos: bool = False


@dataclass
class KVCacheHandle:
    """
    Identifies the KV cache resources owned by a request.

    Attributes:
        slot: The logical batch slot (row) in the KV cache tensors.
        pages: Optional list of page identifiers when paged KV caching is used.
    """

    slot: int
    pages: Optional[List[int]] = None

    def is_paged(self) -> bool:
        return self.pages is not None


@dataclass
class RequestState:
    """
    Encapsulates all mutable state required to service a single generation
    request within the engine.
    """

    request_id: str
    prompt_token_ids: torch.Tensor
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    arrival_time: float = field(default_factory=time.time)

    status: RequestStatus = RequestStatus.WAITING
    kv_cache: Optional[KVCacheHandle] = None
    max_new_tokens: Optional[int] = None  # per request budget

    # Runtime state
    generated_token_ids: List[int] = field(default_factory=list)  # can be used for streaming and decoding at the end
    last_token: Optional[int] = None
    total_tokens_emitted: int = 0

    def prompt_length(self) -> int:
        return int(self.prompt_token_ids.size(-1))

    def set_status(self, status: RequestStatus) -> None:
        self.status = status

    def attach_kv_cache(self, handle: KVCacheHandle) -> None:
        self.kv_cache = handle

    def mark_prefill_complete(self, next_input_token: torch.Tensor) -> None:
        """
        Called once prefill has finished and the request is ready to decode.
        """
        self.status = RequestStatus.DECODE
        # Store the next decode input (shape [B, 1]); keep scalar copy for book-keeping
        self.last_token = int(next_input_token.view(-1)[0])

    def append_token(self, token_id: torch.Tensor) -> None:
        """
        Record a sampled token for this request.
        """
        token_value = int(token_id)
        self.generated_token_ids.append(token_value)
        self.last_token = token_value
        self.total_tokens_emitted += 1

    def should_stop(self) -> bool:
        """
        Determine whether the request has reached a stopping condition.
        """
        if self.status == RequestStatus.FINISHED:
            return True
        if self.max_new_tokens is not None and self.total_tokens_emitted >= self.max_new_tokens:
            return True
        return False

    def mark_finished(self) -> None:
        self.status = RequestStatus.FINISHED

