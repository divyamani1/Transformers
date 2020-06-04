from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchtext import data, datasets
from classifier import TransformerClassifier
from seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args):
    if args.model == "classifier":

        TEXT = data.Field(lower=True, batch_first=True)
        LABEL = data.Field(lower=True, sequential=False)

        train, valid, test = datasets.SST.splits(text_field=TEXT, label_field=LABEL)
        train_iter, valid_iter, test_iter = data.BucketIterator.splits(
            (train, valid, test), batch_size=args.batch_size, device=device
        )

        TEXT.build_vocab(train)
        LABEL.build_vocab(train)

        ntokens = len(TEXT.vocab)
        seq_len = 30

        print(f"Dataset loaded. Vocabulary: {ntokens}\n")

        num_classes = 3

        model = TransformerClassifier(
            ntokens=ntokens,
            embed_size=args.embed_size,
            seq_len=seq_len,
            num_classes=num_classes,
            attn_heads=args.attn_heads,
            depth=args.depth,
            pos_emb=args.positional,
        ).to(device)

        print("Classifier Model Initialized.\n")

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        for e in range(args.num_epochs):
            epoch_loss = 0
            total_loss = 0
            for i, batch in enumerate(tqdm.tqdm(train_iter)):
                optimizer.zero_grad()
                out = model(batch.text)
                loss = F.nll_loss(out, batch.label - 1, reduction="mean")

                total_loss += loss.item()
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Loss at epoch {e+1} is {epoch_loss / len(train_iter)}\n")

            with torch.no_grad():
                model.eval()
                total_loss = 0
                total_acc = 0
                for i, batch in enumerate(tqdm.tqdm(valid_iter)):

                    out = model(batch.text)

                    loss = F.nll_loss(out, batch.label - 1, reduction="mean")
                    total_loss += loss.item()

                    preds = out.argmax(dim=1) + 1

                    total = (preds == batch.label).flatten().size(0)
                    correct = (preds == batch.label).float().sum()

                    val_accuracy = correct / total
                    total_acc += val_accuracy.item()

                print(
                    f"Average Validation Loss={total_loss / len(valid_iter)} Accuracy={total_acc / len(valid_iter)}\n"
                )

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_acc = 0
            for i, batch in enumerate(tqdm.tqdm(test_iter)):
                out = model(batch.text)

                loss = F.nll_loss(out, batch.label - 1)
                total_loss += loss.item()

                preds = out.argmax(dim=1) + 1

                total = (preds == batch.label).flatten().size(0)
                correct = (preds == batch.label).float().sum()

                test_accuracy = correct / total
                total_acc += test_accuracy.item()

            print(
                f"Average Test Loss={total_loss / len(test_iter)} Accuracy={total_acc / len(test_iter)}\n"
            )

    elif args.model == "generator":

        TEXT = data.Field(lower=True, batch_first=True)

        train, valid, test = datasets.WikiText2.splits(TEXT)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test),
            batch_size=args.batch_size,
            bptt_len=30,
            device=device,
            repeat=False,
        )

        TEXT.build_vocab(train)

        ntokens = len(TEXT.vocab)
        seq_len = 30

        print(f"WikiText2 dataset loaded. Vocabulary: {ntokens}\n")
        model = Seq2Seq(
            ntokens=ntokens, embed_size=args.embed_size, seq_len=seq_len
        ).to(device)

        print("Sequence to sequence model initialized.\n")

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        for e in range(args.num_epochs):
            epoch_loss = 0
            total_loss = 0
            for i, batch in enumerate(tqdm.tqdm(train_iter)):
                optimizer.zero_grad()
                out = model(batch.text)
                loss = F.nll_loss(out.transpose(2, 1), batch.target, reduction="mean")

                total_loss += loss.item()
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Loss at epoch {e} is {epoch_loss / len(train_iter)}\n")

            with torch.no_grad():
                model.eval()

                total_loss = 0
                total_acc = 0

                for i, batch in enumerate(tqdm.tqdm(valid_iter)):

                    out = model(batch.text)

                    loss = F.nll_loss(
                        out.transpose(2, 1), batch.target, reduction="mean"
                    )
                    total_loss += loss.item()

                    preds = out.argmax(dim=2)

                    total = (preds == batch.target).flatten().size(0)
                    correct = (preds == batch.target).float().sum()

                    val_accuracy = correct / total
                    total_acc += val_accuracy.item()

                print(
                    f"Average Validation Loss = {total_loss / len(valid_iter)} Accuracy = {total_acc / len(valid_iter)}\n"
                )

        with torch.no_grad():
            model.eval()

            total_loss = 0
            total_acc = 0

            for i, batch in enumerate(tqdm.tqdm(test_iter)):
                out = model(batch.text)

                loss = F.nll_loss(out.transpose(2, 1), batch.target)
                total_loss += loss.item()

                preds = out.argmax(dim=2)

                total = (preds == batch.target).flatten().size(0)
                correct = (preds == batch.target).float().sum()

                test_accuracy = correct / total
                total_acc += test_accuracy.item()

            print(
                f"Average Test Loss = {total_loss / len(test_iter)} Accuracy = {total_acc / len(test_iter)}\n"
            )

    else:
        print("Model not implemented. Available: classifier and generator")
        raise NotImplementedError


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="The model to train: 'classifier' or 'generator'",
        default="classifier",
        type=str,
    )

    parser.add_argument(
        "-N",
        "--num-epochs",
        dest="num_epochs",
        help="The number of epochs for model to train.",
        default=3,
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        dest="lr",
        help="The learning rate.",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "-e",
        "--embed-size",
        dest="embed_size",
        help="The size of the embeddings",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-H",
        "--heads",
        dest="attn_heads",
        help="The number of attention heads.",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size to feed to the model.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        help="The number of transformer blocks.",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-p",
        "--positional",
        dest="positional",
        help="Use positional embeddings.",
        default=False,
        type=bool,
    )

    options = parser.parse_args()

    run(options)
