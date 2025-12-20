# models/TemporalFFRNN.py
import torch
import torch.nn as nn

class TemporalFFRNN(nn.Module):
    """
    (B, T, D) -> per-frame FF -> RNN -> last-timestep -> classifier
    - ff_hidden2: None => 1 FF layer, else 2 FF layers
    - rnn_type: "gru" or "lstm"
    - rnn_layers: 1 or 2
    """
    def __init__(self,
                 input_dim: int,
                 ff_hidden: int,
                 num_classes: int,
                 ff_hidden2: int | None = None,
                 dropout: float = 0.0,
                 rnn_type: str = "gru",
                 rnn_hidden: int = 128,
                 rnn_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        assert rnn_type in ("gru", "lstm")

        ff = [nn.Linear(input_dim, ff_hidden), nn.ReLU()]
        if dropout and dropout > 0: ff.append(nn.Dropout(dropout))
        if ff_hidden2 is not None and ff_hidden2 > 0:
            ff += [nn.Linear(ff_hidden, ff_hidden2), nn.ReLU()]
            if dropout and dropout > 0: ff.append(nn.Dropout(dropout))
            ff_out = ff_hidden2
        else:
            ff_out = ff_hidden
        self.ff = nn.Sequential(*ff)

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        hidden = rnn_hidden
        num_dir = 2 if bidirectional else 1

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=ff_out,
                hidden_size=hidden,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.LSTM(
                input_size=ff_out,
                hidden_size=hidden,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=bidirectional
            )

        self.head = nn.Linear(hidden * num_dir, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        x = self.ff(x)  # (B*T, F)
        F = x.shape[-1]
        x = x.reshape(B, T, F)  # (B, T, F)

        out, _ = self.rnn(x)    # (B, T, H * num_dir)
        last = out[:, -1, :]    # last timestep
        logits = self.head(last)
        return logits

