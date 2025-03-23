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
            dropout: float = 0.0,
            resnet_model: str = 'microsoft/resnet-34'
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        
        # Use a single linear projection as in BaselineModel.
        self.proj = nn.Linear(512, len(tokenizer))
        
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.lstm_layers = lstm_layers

    def forward(self, img, target_seq=None, teacher_forcing=False, detach_loop=False):
        batch_size = img.shape[0]
        # Extract visual features in the same way as the baseline.
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, batch, 512)
        
        # Initialize with SOS token embedding.
        start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
        start_embed = self.embed(torch.tensor(start_idx, device=img.device))  # (512)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # (1, batch, 512)
        
        # Initialize LSTM hidden and cell states.
        # For multi-layer LSTM, repeat the visual feature for each layer.
        hidden = feat.repeat(self.lstm_layers, 1, 1)  # (num_layers, batch, 512)
        cell = torch.zeros_like(hidden)  # (num_layers, batch, 512)
        
        # Append the SOS embedding as the first output.
        outputs = [start_embeds]

        if teacher_forcing and target_seq is not None:
            # Expect target_seq of shape (batch, seq_len) with SOS at index 0.
            seq_len = target_seq.shape[1]
            for t in range(1, seq_len):
                # Use ground-truth input embedding at time t.
                inp = self.embed(target_seq[:, t].long()).unsqueeze(0)  # (1, batch, 512)
                out, (hidden, cell) = self.lstm(inp, (hidden, cell))
                outputs.append(out)
                if detach_loop:
                    hidden = hidden.detach()
                    cell = cell.detach()
        else:
            # Autoregressive generation: loop for text_max_len - 1 iterations since SOS is provided.
            inp = start_embeds
            for t in range(self.text_max_len - 1):
                out, (hidden, cell) = self.lstm(inp, (hidden, cell))
                # Optionally detach the output to cut gradient flow.
                last_out = out[-1:] if not detach_loop else out[-1:].detach()
                outputs.append(last_out)
                inp = last_out

        # Concatenate outputs, permute dimensions, and apply the projection layer.
        res = torch.cat(outputs, dim=0)  # (seq_len, batch, 512)
        res = res.permute(1, 0, 2)  # (batch, seq_len, 512)
        res = self.proj(res)  # (batch, seq_len, vocab_size)
        res = res.permute(0, 2, 1)  # (batch, vocab_size, seq_len)
        return res

    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        indices = torch.argmax(logits, dim=1)  # (batch, seq_len)
        return self.tokenizer.batch_decode(indices.tolist())
    


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # Compute attention scores
        q = self.query(query)  # (batch, seq_len, hidden_dim)
        k = self.key(key)      # (batch, seq_len, hidden_dim)
        v = self.value(value)  # (batch, seq_len, hidden_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))  # (batch, seq_len, seq_len)
        attn_weights = self.softmax(scores)  # (batch, seq_len, seq_len)

        context = torch.matmul(attn_weights, v)  # (batch, seq_len, hidden_dim)
        return context, attn_weights


class LSTMWithAttention(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201,
            lstm_layers: int = 3,
            dropout: float = 0.0,
            resnet_model: str = 'microsoft/resnet-34',
            attention_hidden_dim: int = 512  # Attention hidden dimension
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        
        self.attention = Attention(input_dim=512, hidden_dim=attention_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        
        self.proj = nn.Linear(512, len(tokenizer))
        
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.lstm_layers = lstm_layers

    def forward(self, img):
        batch_size = img.shape[0]
        # Extract visual features in the same way as the baseline.
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, batch, 512)
        
        # Apply attention mechanism to the ResNet output.
        context, attn_weights = self.attention(feat, feat, feat)  # Attention across the ResNet features
        
        # Initialize with SOS token embedding.
        start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
        start_embed = self.embed(torch.tensor(start_idx, device=img.device))  # (512)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # (1, batch, 512)
        
        # Initialize LSTM hidden and cell states.
        # For multi-layer LSTM, repeat the visual feature for each layer.
        hidden = context.repeat(self.lstm_layers, 1, 1)  # (num_layers, batch, 512)
        cell = torch.zeros_like(hidden)  # (num_layers, batch, 512)
        
        # Append the SOS embedding as the first output.
        outputs = [start_embeds]

        # Autoregressive generation: loop for text_max_len - 1 iterations since SOS is provided.
        inp = start_embeds
        for t in range(self.text_max_len - 1):
            # Apply attention to the visual features at each step of LSTM
            context, attn_weights = self.attention(feat, feat, feat)  # Attention across the ResNet features
            out, (hidden, cell) = self.lstm(inp, (hidden, cell))
            
            # Optionally detach the output to cut gradient flow.
            last_out = out[-1:]  # (1, batch, 512)
            outputs.append(last_out)
            inp = last_out

        # Concatenate outputs, permute dimensions, and apply the projection layer.
        res = torch.cat(outputs, dim=0)  # (seq_len, batch, 512)
        res = res.permute(1, 0, 2)  # (batch, seq_len, 512)
        res = self.proj(res)  # (batch, seq_len, vocab_size)
        res = res.permute(0, 2, 1)  # (batch, vocab_size, seq_len)
        return res

    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        indices = torch.argmax(logits, dim=1)  # (batch, seq_len)
        return self.tokenizer.batch_decode(indices.tolist())