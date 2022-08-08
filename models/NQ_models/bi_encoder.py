from typing import Tuple

from torch import nn
from torch import Tensor as T


class BiEncoder(nn.Module):
    def __init__(
            self,
            question_model: nn.Module,
            ctx_model: nn.Module,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model

    def forward_question_model(self, ids: T, segments: T, attn_mask: T, representation_token_pos=0) -> (T, T, T):
        sequence_output, pooled_output, hidden_states = self.question_model(
            ids, segments, attn_mask, representation_token_pos=representation_token_pos)
        return sequence_output, pooled_output, hidden_states

    def forward_ctx_model(self, ids: T, segments: T, attn_mask: T, representation_token_pos=0) -> (T, T, T):
        sequence_output, pooled_output, hidden_states = self.ctx_model(
            ids, segments, attn_mask, representation_token_pos=representation_token_pos)
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        representation_token_pos=0) -> Tuple[T, T]:

        _q_seq, q_pooled_out, _q_hidden = self.forward_question_model(
            question_ids,
            question_segments,
            question_attn_mask,
            representation_token_pos=representation_token_pos
        )

        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.forward_ctx_model(
            context_ids,
            ctx_segments,
            ctx_attn_mask,
        )
        return q_pooled_out, ctx_pooled_out
