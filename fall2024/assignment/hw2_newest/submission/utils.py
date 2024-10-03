import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    """
    With size = 4, we will have:
    subsequent_mask = [[0 1 1 1]
                       [0 0 1 1]
                       [0 0 0 1]
                       [0 0 0 0]]
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )

    """
    returns
     [[1 0 0 0]
       [1 1 0 0]
       [1 1 1 0]
       [1 1 1 1]]
    """
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    """
    Implement beam search decoding with 'beam_size' width
    start_symbol: Typically <s> idx 0
        end_symbol </s> idx 1
        unknown <unk> idx 3
        padding <blank> idx 2
    """

    #To-do: Encode source input using the model
    memory = model.encode(src, src_mask) # (beam_size, seq_len, d_model)

    # Initialize decoder input and scores
    # Denote how many sequences have been generated, initialized to be tensor[[0]], just the start token
    # (batch_size * beam_size, seq_len)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    # scores: records the top_k prob for different beans, (batch_size * beam_size, vocab)
    scores = torch.Tensor([0.]).cuda()

    # Generated a max_len for each input sentence
    for i in range(max_len - 1):
        # TODO: Decode using the model, memory, and source mask
        # out (beam_size, seq_len, d_model)
        # subsequent_mask (1, seq_len, seq_len), refer to greedy_decode()
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))  # Replace with model.decode(...)
        # Calculate probabilities for the next token
        # out[:, -1, :] => (beam_size, d_model)
        # prob => (beam_size, vocab_size)
        prob = model.generator(out[:, -1, :])  # Replace
        # Set probabilities of end token to 0 (except when already ended)
        prob[ys[:, -1] == end_idx, :] = 0
        """
        Update scores, basically used to calculate accumulated scores defined as sum of log prob
        scores is (beam_size, ) but prob is (beam_size, vocab_size),
        and we need to sum up the probability since score(y_1, y_2,... y_n) = Σ log p (y_i | y_{<i}),
        need to expand the shape of scores to be (beam_size, 1) to enable broadcasting
        """
        scores = scores.unsqueeze(1) + prob
        # Get top-k scores and indices
        """
        Here scores.reshape(-1) concatenates different beam scores together
        scores = [[-0.1, -0.3, -0.8, ...], [-0.2, -0.4, -0.123, ...]] => scores = [-0.1, -0.3, -0.8, ...,  -0.2, -0.4, ...]
        scores = [-0.1, -0.2] if beam_size = 2
        """
        scores, indices = torch.topk(scores.reshape(-1), beam_size)

        # TODO: Extract beam indices and token indices from top-k scores
        vocab_size = prob.size(-1)
        # calculate which beam to choose for next expansion
        beam_indices = torch.divide(indices, vocab_size, rounding_mode='floor')  # Replace with torch.divide(indices, vocab_size, rounding_mode='floor')
        token_indices = torch.remainder(indices, vocab_size)  # Replace with torch.remainder(indices, vocab_size)

        # Prepare next decoder input
        next_decoder_input = []
        for beam_index, token_index in zip(beam_indices, token_indices):

            # TODO: Handle end token condition and append to next_decoder_input
            curr_decoder_input_seq = ys[beam_index]
            # Early stop, termination condition
            if curr_decoder_input_seq[-1] == end_idx:
                token_index = end_idx

            # torch([[0, 2344, 4, 5, 6]])
            token_index = torch.LongTensor([token_index]).cuda()
            """
            [[0, 2344, 4, 5, 6]] -> [[0, 2344, 4, 5, 6, next_token]]
            """
            next_decoder_input.append(torch.cat([curr_decoder_input_seq, token_index]))

        # Update ys
        """
        torch([[0, 2344, 4, 5, 6, next_token_beam_1],
        [0, 23, 5,6, 648, next_token_beam_2]),
        ...]
        """
        ys = torch.vstack(next_decoder_input)

        # Check if all beams are finished, exit
        eos_count = (ys[:, -1] == end_idx).sum()
        if eos_count == beam_size:
            break

        # Expand encoder output for beam size (only once)
        if i == 0:
            memory = memory.repeat(beam_size, 1, 1)
            src_mask = src_mask.repeat(beam_size, 1, 1)

    # Return the top-scored sequence, topK has sorted the sequence
    ys = ys[0]
    # convert the top scored sequence to a list of text tokens
    ys = ys.unsqueeze(0)

    return ys
        


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for s in batch:
        _src = s['de']
        _tgt = s['en']
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):

    if sent.startswith('<s>'):
        sent = sent[3:]

    if sent.endswith('</s>'):
        sent = sent[:-4]

    return sent


def compute_corpus_level_bleu(refs, hyps):

    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return bleu.score

