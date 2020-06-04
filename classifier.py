import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import TransformerBlock, PositionalEncoding


class TransformerClassifier(nn.Module):
    """
    A simple classifier using Transformer architecture
    """

    def __init__(
        self,
        ntokens,
        embed_size,
        seq_len,
        num_classes=3,
        attn_heads=4,
        depth=1,
        pos_emb=False,
    ):
        """
        :param ntokens: No. of tokens (vocabulary)
        :param embed_size: Embedding size
        :param seq_len: Length of the input sequence
        :param num_classes: No. of classes to predict
        :param attn_heads: No. of attention heads
        :param depth: No. of transformer blocks
        :pos_emb: If true, use learned positional embeddings. If false, use positional encodings.
        """
        super(TransformerClassifier, self).__init__()
        self.ntokens = ntokens

        self.embeddings = nn.Embedding(ntokens, embed_size)

        if pos_emb:
            self.pos_emb = nn.Embedding(seq_len, embed_size)
        else:
            self.pos_emb = PositionalEncoding(embed_size)

        # Define the number of transformer blocks to use
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                TransformerBlock(attn_heads=attn_heads, embed_size=embed_size)
            )

        self.tblocks = nn.Sequential(*transformer_blocks)

        self.toprobs = nn.Linear(embed_size, num_classes)

    def forward(self, x):

        embeds = self.embeddings(x)
        pos = self.pos_emb(x)
        x = embeds + pos

        out = self.tblocks(x)

        out = self.toprobs(out.mean(dim=1))
        return F.log_softmax(out, dim=1)
