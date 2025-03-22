import torch
from torch import nn
from transformers import ResNetModel
from our_tokenizers import CharacterTokenizer


class BaselineModel(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201,
            resnet_model: str = 'microsoft/resnet-18'
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, len(tokenizer))
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def forward(self, img, target_seq=None, teacher_forcing=False, detach_loop=False):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, batch, 512)
        # Initialize with SOS token embedding
        start = torch.tensor(self.tokenizer.char2idx[self.tokenizer.sos_token]).to(img.device)
        start_embed = self.embed(start)  # (512)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # (1, batch, 512)
        
        hidden = feat
        outputs = [start_embeds]  # store the SOS embedding as first output

        if teacher_forcing and target_seq is not None:
            # Expect target_seq of shape (batch, seq_len) with SOS at index 0.
            seq_len = target_seq.shape[1]
            for t in range(1, seq_len):
                # Ground-truth input embedding at time t
                inp = self.embed(target_seq[:, t].long()).unsqueeze(0)  # (1, batch, 512)
                out, hidden = self.gru(inp, hidden)
                outputs.append(out)
                if detach_loop:
                    # Optionally detach the hidden state even when using teacher forcing.
                    hidden = hidden.detach()
        else:
            # Autoregressive generation
            inp = start_embeds
            for t in range(self.text_max_len - 1):  # -1 because SOS is provided
                out, hidden = self.gru(inp, hidden)
                # Depending on detach_loop, choose whether to cut the gradient flow.
                last_out = out[-1:] if not detach_loop else out[-1:].detach()
                outputs.append(last_out)
                inp = last_out

        res = torch.cat(outputs, dim=0)  # (seq_len, batch, 512)
        res = res.permute(1, 0, 2)  # (batch, seq_len, 512)
        res = self.proj(res)  # (batch, seq_len, vocab_size)
        res = res.permute(0, 2, 1)  # (batch, vocab_size, seq_len)
        return res

    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        indices = torch.argmax(logits, dim=1)  # (batch, seq_len)
        texts = [self.tokenizer.decode(seq.tolist()) for seq in indices]
        return texts


class LSTMModel(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201,
            lstm_layers: int = 3,
            dropout: float = 0.3,
            resnet_model: str = 'microsoft/resnet-34'
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False  # (seq_len, batch, features)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, len(tokenizer))
        )
        
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.lstm_layers = lstm_layers
        self.dropout = nn.Dropout(dropout)
        self.visual_proj = nn.Linear(512, 512)

    def forward(self, img, target_seq=None, teacher_forcing=False, detach_loop=False):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output
        feat = feat.squeeze([-1, -2])  # (batch, 512)
        feat = self.visual_proj(feat)
        
        hidden = self._init_hidden(feat, img.device)  # (num_layers, batch, 512)
        cell = torch.zeros_like(hidden).to(img.device)
        
        start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
        inp = torch.full((batch_size,), start_idx, device=img.device)
        inp = self.embed(inp).unsqueeze(0)  # (1, batch, 512)
        
        outputs = []
        if teacher_forcing and target_seq is not None:
            seq_len = target_seq.shape[1]
            for t in range(1, seq_len):
                out, (hidden, cell) = self.lstm(inp, (hidden, cell))
                out = self.dropout(out)
                outputs.append(out)
                # Use ground-truth input embedding at time t
                inp = self.embed(target_seq[:, t].long()).unsqueeze(0)
                if detach_loop:
                    hidden = hidden.detach()
                    cell = cell.detach()
        else:
            for t in range(self.text_max_len):
                out, (hidden, cell) = self.lstm(inp, (hidden, cell))
                out = self.dropout(out)
                outputs.append(out)
                inp = out[-1:] if not detach_loop else out[-1:].detach()

        res = torch.cat(outputs, dim=0)  # (seq_len, batch, 512)
        res = res.permute(1, 0, 2)  # (batch, seq_len, 512)
        return self.proj(res).permute(0, 2, 1)  # (batch, vocab_size, seq_len)

    def _init_hidden(self, feat: torch.Tensor, device: torch.device) -> torch.Tensor:
        hidden = feat.unsqueeze(0)  # (1, batch, 512)
        if self.lstm_layers > 1:
            hidden = hidden.repeat(self.lstm_layers, 1, 1)
            hidden = torch.stack([self.visual_proj(hidden[i]) for i in range(self.lstm_layers)])
        return hidden

    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        indices = torch.argmax(logits, dim=1)  # (batch, seq_len)
        return self.tokenizer.batch_decode(indices.tolist())
