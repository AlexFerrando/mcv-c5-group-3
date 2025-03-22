import torch
from torch import nn
from transformers import ResNetModel
from our_tokenizers import CharacterTokenizer


class BaselineModel(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18')
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, len(tokenizer))
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        start = torch.tensor(self.tokenizer.char2idx[self.tokenizer.sos_token]).to(img.device)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        for t in range(self.text_max_len - 1): # -1 because we already have the <SOS> token
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512
    
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80
        res = res.permute(0, 2, 1) # batch, 80, seq
        return res
    
    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        """
        Converts model logits to text sequences.
        Args:
            logits: (batch, vocab_size, sequence_length)
        Returns:
            List of decoded strings.
        """
        indices = torch.argmax(logits, dim=1)  # (batch, sequence_length)
        texts = [self.tokenizer.decode(seq.tolist()) for seq in indices]
        return texts

class LSTMModel(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201,
            lstm_layers: int = 3,       # New parameter
            dropout: float = 0.3        # New parameter
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18')
        
        # Enhanced LSTM configuration
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,  # Dropout only between layers
            batch_first=False  # Maintain (seq_len, batch, features) format
        )
        
        # Additional projection layers
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

        # Visual feature projection layer
        self.visual_proj = nn.Linear(512, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        
        # Extract visual features
        feat = self.resnet(img).pooler_output
        feat = feat.squeeze([-1, -2])  # (batch, 512)
        feat = self.visual_proj(feat)  # Project to latent space
        
        # Initialize states for multiple layers
        hidden = self._init_hidden(feat, img.device)  # (num_layers, batch, 512)
        cell = torch.zeros_like(hidden).to(img.device)
        
        # Initial sequence
        start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
        inp = torch.full((batch_size,), start_idx, device=img.device)
        inp = self.embed(inp).unsqueeze(0)  # (1, batch, 512)
        
        outputs = []
        for t in range(self.text_max_len - 1):
            # LSTM forward pass
            out, (hidden, cell) = self.lstm(inp, (hidden, cell))
            
            # Apply dropout
            out = self.dropout(out)
            
            # Store output
            outputs.append(out)
            
            # Prepare next input (autoregressive)
            inp = out[-1:].detach()  # Detach for numerical stability

        # Concatenate all outputs
        res = torch.cat(outputs, dim=0)  # (seq_len, batch, 512)
        
        # Final projection
        res = res.permute(1, 0, 2)  # (batch, seq_len, 512)
        return self.proj(res).permute(0, 2, 1)  # (batch, vocab_size, seq_len)

    def _init_hidden(self, feat: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for all LSTM layers"""
        # feat shape: (batch, 512)
        hidden = feat.unsqueeze(0)  # (1, batch, 512)
        
        # Handle multiple layers
        if self.lstm_layers > 1:
            hidden = hidden.repeat(self.lstm_layers, 1, 1)  # (num_layers, batch, 512)
            # Optional: Layer-specific projection
            hidden = torch.stack([
                self.visual_proj(hidden[i]) for i in range(self.lstm_layers)
            ])
        
        return hidden

    def logits_to_text(self, logits: torch.Tensor) -> list[str]:
        """Convert model logits to text sequences.
        
        Args:
            logits: (batch, vocab_size, sequence_length)
        Returns:
            List of decoded strings
        """
        indices = torch.argmax(logits, dim=1)
        return [self.tokenizer.decode(seq.tolist()) for seq in indices]
