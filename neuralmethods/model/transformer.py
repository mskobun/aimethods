import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def creatMask(batch, sequence_length):
    mask = torch.zeros(batch, sequence_length, sequence_length)
    for i in range(sequence_length):
        mask[:, i, : i + 1] = 1
    return mask


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None, returnWeights=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # print("Scores in attention itself",torch.sum(scores))
    if returnWeights:
        return output, scores

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.h = heads

        # Calculate padded dimension to make it divisible by heads
        self.pad = (heads - (d_model % heads)) % heads  # Padding needed
        self.padded_d_model = d_model + self.pad  # Padded dimension
        self.d_k = self.padded_d_model // heads  # Features per head

        # Linear layers now work with padded dimensions
        self.q_linear = nn.Linear(d_model, self.padded_d_model)
        self.v_linear = nn.Linear(d_model, self.padded_d_model)
        self.k_linear = nn.Linear(d_model, self.padded_d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(
            self.padded_d_model, d_model
        )  # Map back to original dimension

    def forward(self, q, k, v, mask=None, returnWeights=False):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        if returnWeights:
            scores, weights = attention(
                q, k, v, self.d_k, mask, self.dropout, returnWeights=returnWeights
            )
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.padded_d_model)
        output = self.out(concat)  # Map back to original dimension

        if returnWeights:
            return output, weights
        else:
            return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=400, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnWeights=False):
        x2 = self.norm_1(x)
        # print(x2[0,0,0])
        # print("attention input.shape",x2.shape)
        if returnWeights:
            attenOutput, attenWeights = self.attn(
                x2, x2, x2, mask, returnWeights=returnWeights
            )
        else:
            attenOutput = self.attn(x2, x2, x2, mask)
        # print("attenOutput",attenOutput.shape)
        x = x + self.dropout_1(attenOutput)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        if returnWeights:
            return x, attenWeights
        else:
            return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * (i + 1)) / d_model))
                    )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Check if input is transposed (batch, feature, seq) or (batch, seq, feature)
        # For transposed input, we need to transpose back for positional encoding
        transposed_input = x.size(1) == self.d_model

        if transposed_input:
            x = x.transpose(1, 2)  # Convert to (batch, seq, feature)

        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        x = self.dropout(x)

        # Transpose back if input was transposed
        if transposed_input:
            x = x.transpose(1, 2)  # Convert back to (batch, feature, seq)

        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, input_size, seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.input_size = input_size
        self.pe = PositionalEncoder(input_size, seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(input_size, heads, dropout), N)
        self.norm = Norm(input_size)

    def forward(self, x, mask=None, returnWeights=False):
        # Check if input is transposed (batch, feature, seq) or (batch, seq, feature)
        transposed_input = x.size(1) == self.input_size

        # Apply positional encoding
        x = self.pe(x)

        # Process through transformer layers
        for i in range(self.N):
            if i == 0 and returnWeights:
                x, weights = self.layers[i](x, mask=mask, returnWeights=returnWeights)
            else:
                x = self.layers[i](x, mask=mask)

        # Apply normalization
        x = self.norm(x)

        if returnWeights:
            return x, weights
        else:
            return x


class Transformer(nn.Module):
    def __init__(
        self, n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub
    ):
        super().__init__()
        self.encoder = Encoder(n_feature, n_timestep, n_layer, n_head, n_dropout)
        self.out = nn.Linear(n_feature, n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)
        self.lb = lb
        self.ub = ub

    def forward(self, src, returnWeights=False):
        mask = creatMask(src.shape[0], src.shape[1]).to(device)
        # print(src.shape)
        if returnWeights:
            e_outputs, weights, z = self.encoder(src, mask, returnWeights=returnWeights)
        else:
            e_outputs = self.encoder(src, mask)

        e_outputs = self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        output = self.out(e_outputs)
        output = F.softmax(output, dim=1)
        output = torch.stack(
            [self.rebalance(batch, self.lb, self.ub) for batch in output]
        )
        if returnWeights:
            return output, weights
        else:
            return output

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped
