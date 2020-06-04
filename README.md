# Transformers
This is a basic implementation of the transformer architecture from the paper ["Attention is all you need"](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). The repository includes a basic transformer architecture with multi-headed self attention and positional encodings. The `classifier.py` and `seq2seq.py` show two application of the implementation. The implementation of the transformer block is in the `modules.py` file.

# Running the model
First, clone the repository with,

`git clone https://github.com/divyamani1/Transformers`

`cd Transformers/`

Then, install the requirements with,

`pip install -r  requirements.txt`

```
usage: run.py [-h] [-m MODEL] [-N NUM_EPOCHS] [-lr LR] [-e EMBED_SIZE]
              [-H ATTN_HEADS] [-b BATCH_SIZE] [-d DEPTH] [-p POSITIONAL]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model to train: 'classifier' or 'generator'
  -N NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        The number of epochs for model to train.
  -lr LR, --learning-rate LR
                        The learning rate.
  -e EMBED_SIZE, --embed-size EMBED_SIZE
                        The size of the embeddings
  -H ATTN_HEADS, --heads ATTN_HEADS
                        The number of attention heads.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to feed the input sequences to the model.
  -d DEPTH, --depth DEPTH
                        The number of transformer blocks.
  -p POSITIONAL, --positional POSITIONAL
                        If true, use positional embeddings. If false, use positional encodings.
```
# References
[Transformers from scratch](http://www.peterbloem.nl/blog/transformers)
