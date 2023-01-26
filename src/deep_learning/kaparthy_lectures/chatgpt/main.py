import torch
import torch.nn as nn

from deep_learning.kaparthy_lectures.chatgpt.model import BiagramLanguageModel

torch.manual_seed(1337)


def load_data():
    with open("custom_data/input.txt", "r", encoding="utf-8") as f:
        data = f.read()
    return data


class DataHandler:
    def __init__(
        self, data: str, train_val_split: float = 0.9, block_size: int = 8, batch_size: int = 4, device: str = "cuda"
    ) -> None:

        # extract all the unique characters
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)

        # mapping between chars and integers and vice versa
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        self.encode = lambda x: [stoi[c] for c in x]  # encoding: converting characters to a list of integers
        self.decode = lambda x: "".join([itos[c] for c in x])  # decoding: converting integers to list of characters

        self.data = torch.tensor(self.encode(data), dtype=torch.long)

        split_idx = int(train_val_split * len(self.data))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_random_batch(self, data_type: str):
        data = self.train_data if data_type == "train" else self.val_data

        # get a random starting point
        idx = torch.randint(0, len(data) - self.block_size, size=(self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in idx])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class Trainer:
    def __init__(
        self,
        model: BiagramLanguageModel,
        dhandler: DataHandler,
        eval_interval: int,
        eval_iters: int,
        train_iters: int,
        device: str = "cuda",
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.model = model
        self.dhandler = dhandler
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.train_iters = train_iters
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device

        # move model to device
        self.model.to(self.device)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.dhandler.get_random_batch(split)
                loss, _ = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train_iteration(self):
        X, Y = self.dhandler.get_random_batch("train")
        loss, logits = self.model(X, Y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def train(self):
        for k in range(self.train_iters):
            self.train_iteration()
            if k % self.eval_interval == 0:
                loss = self.estimate_loss()
                # print(f"train loss: {loss['train']:.3f}, val loss: {loss['val']:.3f}")
                print(f"step {k}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")


def main(
    batch_size: int = 32,
    block_size: int = 8,
    device: str = "cuda",
    eval_iters: int = 300,
    train_iters: int = 6000,
    eval_interval: int = 300,
    additional_predicted_chars: int = 400,
    lr: float = 1e-3,
    # Transformer relevant
    n_embeddings: int = 32,
    head_size: int = 16,
    num_heads=4,
    n_layer=3,
):
    dhandler = DataHandler(load_data(), batch_size=batch_size, block_size=block_size, device=device)
    model = BiagramLanguageModel(
        vocab_size=dhandler.vocab_size,
        n_embeddings=n_embeddings,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        n_layer=n_layer,
    )

    trainer = Trainer(
        model,
        dhandler,
        eval_iters=eval_iters,
        train_iters=train_iters,
        eval_interval=eval_interval,
        device=device,
        lr=lr,
    )
    trainer.train()

    # generate some sample text starting with a new line
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    pred = trainer.model.predict(idx=context, additional_predicted_chars=additional_predicted_chars).tolist()

    print(dhandler.decode(pred[0]))


if __name__ == "__main__":
    # main()
    main(
        batch_size=64,
        block_size=256,
        lr=3e-4,
        n_embeddings=384,
        num_heads=6,
        n_layer=6,
        eval_iters=500,
        train_iters=5000,
        eval_interval=200,
        additional_predicted_chars=10000,
    )
