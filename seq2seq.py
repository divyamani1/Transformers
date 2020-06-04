import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import TransformerBlock, PositionalEncoding
from utils.utils import subsequent_mask


class Encoder(nn.Module):
    """
    A simple encoder architecture using multiple transformer blocks
    """

    def __init__(
        self, ntokens, embed_size, seq_len, attn_heads=4, depth=1, pos_emb=False
    ):
        """
        :param ntokens: No. of tokens (vocabulary)
        :param embed_size: Embedding size
        :param seq_len: Length of the input sequence
        :param attn_heads: No. of attention heads
        :param depth: No. of transformer blocks
        :pos_emb: If true, use learned positional embeddings. If false, use positional encodings.
        """
        super(Encoder, self).__init__()
        self.ntokens = ntokens

        self.embeddings = nn.Embedding(ntokens, embed_size)

        if pos_emb:
            self.pos_emb = nn.Embedding(seq_len, embed_size)
        else:
            self.pos_emb = PositionalEncoding(embed_size)

        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                TransformerBlock(attn_heads=attn_heads, embed_size=embed_size)
            )

        self.tblocks = nn.Sequential(*transformer_blocks)

    def forward(self, x):
        """
        :param x: Input sequence to the encoder
        :return: Encoded representation of input sequence
        """
        embeds = self.embeddings(x)
        pos_embeds = self.pos_emb(x)
        inp = embeds + pos_embeds

        out = self.tblocks(inp)

        return out


class DecoderLayer(nn.Module):
    """
    A simple decoder layer
    """

    def __init__(self, ntokens, embed_size, seq_len, attn_heads=4, pos_emb=False):
        """
        :param ntokens: No. of tokens (vocabulary)
        :param embed_size: Embedding size
        :param seq_len: Length of the input sequence
        :param attn_heads: No. of attention heads
        :pos_emb: If true, use learned positional embeddings. If false, use positional encodings.
        """
        super(DecoderLayer, self).__init__()
        self.ntokens = ntokens

        self.embeddings = nn.Embedding(ntokens, embed_size)

        if pos_emb:
            self.pos_emb = nn.Embedding(seq_len, embed_size)
        else:
            self.pos_emb = PositionalEncoding(embed_size)

        # Transformer with masked self-attention
        self.masked_transformer = TransformerBlock(
            attn_heads=attn_heads, embed_size=embed_size
        )

        self.transformer = TransformerBlock(
            attn_heads=attn_heads, embed_size=embed_size
        )

    def forward(self, x, enc_out):
        """
        :param x: Input sequence to decoder
        :param enc_out: Output representation from the encoder
        :return: Representation from decoder
        """
        embeds = self.embeddings(x)
        pos_embeds = self.pos_emb(x)
        inp = embeds + pos_embeds

        mask = subsequent_mask(shape=(1, inp.size(1), inp.size(1)))

        masked_out = self.masked_transformer(inp, mask=mask)

        out = self.transformer(masked_out, mask=None, enc_out=enc_out)
        return out


class Decoder(nn.Module):
    """
    A simple decoder architecture. Currently, it uses a single transformer block with masking.
    """

    def __init__(
        self, ntokens, embed_size, seq_len, attn_heads=4, depth=1, pos_emb=False
    ):
        """
        :param ntokens: No. of tokens (vocabulary)
        :param embed_size: Embedding size
        :param seq_len: Length of the input sequence
        :param attn_heads: No. of attention heads
        :param depth: No. of transformer blocks (currently only one is used)
        :pos_emb: If true, use learned positional embeddings. If false, use positional encodings.
        """
        super(Decoder, self).__init__()

        self.decoder = DecoderLayer(
            ntokens, embed_size, seq_len, attn_heads=attn_heads, pos_emb=pos_emb
        )

    def forward(self, x, enc_out):
        """
        :param x: Input sequence
        :param enc_out: Output from encoder
        :return: Output representation from decoder
        """
        out = self.decoder(x=x, enc_out=enc_out)
        return out


class Seq2Seq(nn.Module):
    """
    A sequence to sequence model using transformer architecture
    """

    def __init__(self, ntokens, embed_size, seq_len):
        """
        :param ntokens: No. of tokens (vocabulary)
        :param embed_size: Embedding size
        :param seq_len: Length of the input sequence
        """
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(
            ntokens, embed_size, seq_len, attn_heads=4, depth=2, pos_emb=False
        )
        self.decoder = Decoder(
            ntokens, embed_size, seq_len, attn_heads=4, depth=2, pos_emb=False
        )

        self.fc = nn.Linear(embed_size, ntokens)

    def forward(self, x):
        """
        :param x: Input sequence to model
        :return: Predictions from the model
        """
        enc_out = self.encoder(x)
        out = self.decoder(x, enc_out=enc_out)
        out = self.fc(out)
        return F.log_softmax(out, dim=2)
