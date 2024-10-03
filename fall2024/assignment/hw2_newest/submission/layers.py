import torch
import torch.nn as nn
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # KQV Self Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # KQV Cross Attention, query is from decoder side, key and value are from encoder output
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    # Your code here
    """
    input:
    @param query: (batch_size, (num_head), seq_len, d_q)
    @param key: (batch_size, (num_head), seq_len, d_k)
    @param value: (batch_size, (num_head), seq_len, d_v)
    @param mask: (batch_size, (num_head), seq_len) of 1 and 0 where 1 means
    @param dropout: A predefined layer that apply dropout to the attention weights

    returns:
    - attention_out: (batch_size, seq_len, d_v)
    - attention_weights: (batch_size, seq_len, seq_len)
    """
    # Step 1: Get the query/key embedding dim
    d_k = query.size(-1)

    # Step 2: Apply softmax(QK.T/sqrt{d_k}) to get the attn_weights ()
    # Step 2.1 Calculate the dot product (batch_size, seq_len, seq_len)
    batch_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2.2 Apply the mask so that the q_i * k_j is zero where mask[:, j] is 0
    if mask is not None:
        # broadcast along seq_len, since we are masking out along the last dimension
        mask = mask.unsqueeze(1)
        batch_scores = batch_scores.masked_fill_(mask == 0, float("-inf"))
    # Step 2.3 Calculate attention weights with softmax
    attention_weights = batch_scores.softmax(dim=-1)  # (batch_size, seq_len, seq_len)

    # Step 3: Apply dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 4: Aggregate the weighted sum of value vectors
    attention_out = torch.matmul(attention_weights, value)
    return attention_out, attention_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # Your code here
        """
        Input:
        @param h: number of heads to use
        @param d_model: d_k * h, used for d_q = d_k, d_v
        @param dropout: layer to dropout attention
        """
        super().__init__()
        self.num_head = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.dropout = nn.Dropout(p=dropout)
        # Wq, Wk, Wv, Wo projections
        self.projections = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        # Your code here
        """
        Inputs:
        - query/key/value: (batch_size, seq_len, d_model), d_model = h * d_k
        - mask: (batch_size, d_model)
        """

        # Step 1: Calculate Attention(QWq, KWk, VWv)
        kqv_packed = [query, key, value]
        batch_size = query.size(0)
        [query_input, key_input, value_input] = [
            projection(mat).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
            for projection, mat
            in zip(self.projections[:-1], kqv_packed)]
        # here query_input/key_input are (batch_size, num_head, seq_len, d_k)

        # Step 2: Calculate multi-headed attention score
        attention_out, self.attn = attention(query_input, key_input, value_input
                                             , mask=mask, dropout=self.dropout)
        # (batch_size, num_head, seq_len, d_v), (batch_size, num_head, seq_len, seq_len)

        # Step 3: Perform concatenation
        # Need Contiguous to make sure view() works fine
        attention_out = attention_out.transpose(1, 2) \
            .contiguous() \
            .view(batch_size, -1, self.d_model)
        return self.projections[-1](attention_out)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_ff here is used for up_sampling
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
