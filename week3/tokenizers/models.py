import torch
from torch import nn
from transformers import ResNetModel

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from our_tokenizers import CharacterTokenizer

class BaselineModel(nn.Module):
    def __init__(
            self,
            tokenizer: CharacterTokenizer,
            text_max_len: int = 201,
            resnet_model: str = 'microsoft/resnet-18',
            start_idx=None
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, len(tokenizer))
        self.embed = nn.Embedding(len(tokenizer), 512)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

        if start_idx is not None:
            self.start_idx = start_idx
        else:
            if hasattr(self.tokenizer, 'char2idx'):
                self.start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
            elif hasattr(self.tokenizer, 'word2idx'):
                self.start_idx = self.tokenizer.word2idx[self.tokenizer.sos_token]
            elif hasattr(self.tokenizer, 'tokenizer'):
                self.start_idx = self.tokenizer.tokenizer.cls_token_id or self.tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError("Unable to determine start index for the tokenizer.")


    def forward(self, img, target_seq=None, teacher_forcing=False, detach_loop=False):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        
        start = torch.tensor(self.start_idx).to(img.device)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512

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
            lstm_layers: int = 3,
            dropout: float = 0.3,
            resnet_model: str = 'microsoft/resnet-34',
            start_idx = None
        ):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(resnet_model)
        
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

        if start_idx is not None:
            self.start_idx = start_idx
        else:
            if hasattr(self.tokenizer, 'char2idx'):
                self.start_idx = self.tokenizer.char2idx[self.tokenizer.sos_token]
            elif hasattr(self.tokenizer, 'word2idx'):
                self.start_idx = self.tokenizer.word2idx[self.tokenizer.sos_token]
            elif hasattr(self.tokenizer, 'tokenizer'):
                self.start_idx = self.tokenizer.tokenizer.cls_token_id or self.tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError("Unable to determine start index for the tokenizer.")

    def forward(self, img, target_seq=None, teacher_forcing=False, detach_loop=True):
        assert not teacher_forcing, "Teacher forcing not supported for this model."
        assert detach_loop, "Detaching the loop is required for this model."
        if target_seq is not None:
            raise ValueError("Target sequence should not be provided for this model.")

        batch_size = img.shape[0]
        
        # Extract visual features
        feat = self.resnet(img).pooler_output
        feat = feat.squeeze([-1, -2])  # (batch, 512)
        feat = self.visual_proj(feat)  # Project to latent space
        
        # Initialize states for multiple layers
        hidden = self._init_hidden(feat, img.device)  # (num_layers, batch, 512)
        cell = torch.zeros_like(hidden).to(img.device)
        
        # Initial sequence
        start_id = torch.tensor(self.start_idx).to(img.device)
        inp = torch.full((batch_size,), start_id, device=img.device)
        inp = self.embed(inp).unsqueeze(0)  # (1, batch, 512)
        
        outputs = []
        for t in range(self.text_max_len):
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
        indices = torch.argmax(logits, dim=1)  # (batch, sequence_length)
    
        return self.tokenizer.batch_decode(indices.tolist())
    