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
        start = torch.tensor(self.tokenizer.encode('<SOS>'))
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