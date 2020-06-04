import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    A single block of Transformer
    """

    def __init__(self, attn_heads, embed_size):
        """
        :param attn_heads: No. of attention heads
        :param embed_size: Embedding size
        """
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.h = attn_heads

        # Multihead attention
        self.attention = MultiHeadSelfAttention(
            embed_size=self.embed_size, attn_heads=self.h
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        # Linear layers for feed-forward network
        self.lin1 = nn.Linear(embed_size, 4 * embed_size)
        self.lin2 = nn.Linear(4 * embed_size, embed_size)

    def forward(self, x, mask=None, enc_out=None):
        # Transformer sub-block: Multi-head attention and normalization
        attn_scores = self.attention(x, mask=mask, enc_out=enc_out)

        normalized_attn = self.norm1(attn_scores + x)

        # Transformer sub-block: Feed-forward network and normalization
        out = self.lin1(normalized_attn)
        out = F.relu(out)
        out = self.lin2(out)
        out = self.norm2(out + normalized_attn)

        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self attention to calculate attention score for input sequence
    """

    def __init__(self, embed_size, attn_heads=4):
        """
        :param embed_size: Embedding size
        :param attn_heads: No. of attention heads
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.attn_heads = attn_heads

        # Queries, Keys and values representation from input tensor
        self.queries = nn.Linear(embed_size, attn_heads * embed_size, bias=False)
        self.keys = nn.Linear(embed_size, attn_heads * embed_size, bias=False)
        self.values = nn.Linear(embed_size, attn_heads * embed_size, bias=False)

        # Combining output for all attention heads
        self.combined = nn.Linear(attn_heads * embed_size, embed_size, bias=False)

    def forward(self, X, mask=None, enc_out=None):
        # Get the batch size, sequence length and embedding size from the input
        batch_size, seq_len, embed_size = X.size()

        # number of attention heads
        h = self.attn_heads

        # If previous encoder output is available, initialize queries and keys from it
        if enc_out is not None:
            queries = self.queries(enc_out).view(batch_size, seq_len, h, embed_size)
            keys = self.keys(enc_out).view(batch_size, seq_len, h, embed_size)
        else:
            queries = self.queries(X).view(batch_size, seq_len, h, embed_size)
            keys = self.keys(X).view(batch_size, seq_len, h, embed_size)

        values = self.values(X).view(batch_size, seq_len, h, embed_size)

        queries = (
            queries.transpose(1, 2)
            .contiguous()
            .view(batch_size * h, seq_len, embed_size)
        )
        keys = (
            keys.transpose(1, 2).contiguous().view(batch_size * h, seq_len, embed_size)
        )
        values = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size * h, seq_len, embed_size)
        )

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot_scaled = dot / (embed_size ** (1 / 2))

        if mask is not None:
            # Apply mask, if required
            dot_scaled = dot_scaled.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot_scaled, dim=2)

        out = torch.bmm(dot, values).view(batch_size, h, seq_len, embed_size)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, h * embed_size)

        # Attention scores
        scores = self.combined(out)

        return scores


class PositionalEncoding(nn.Module):
    """
    Positional encoding to encode positional data for the input sequence.
    """

    def __init__(self, embed_size, max_len=5000):
        """
        :param embed_size: Size of the embeddings
        :param max_len: Maximum length for positional encoding
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.size(0), :]
