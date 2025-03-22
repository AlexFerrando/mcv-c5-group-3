from torchtune.modules.tokenizers._utils import BaseTokenizer
from typing import List, Dict, Any

class CharacterTokenizer(BaseTokenizer):
    def __init__(
            self,
            text_max_len: int = 201
        ):
        super().__init__()
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.chars = [self.sos_token, self.eos_token, self.pad_token, '\n', ' ', '!', '"', '#', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '9', ':', ';', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\x92', '\x96', '\xa0', '®', 'Á', 'É', 'à', 'á', 'â', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ō', 'ơ', '̀', '́', '̃', '̉', 'С', 'и', 'к', 'н', 'р', 'ы', '\u2009', '–', '—', '‘', '’', '“', '”', '강', '개', '닭', '된', '장', '전', '정', '찌', '파']
        self.idx2char = {k: v for k, v in enumerate(self.chars)}
        self.char2idx = {v: k for k, v in enumerate(self.chars)}
        self.text_max_len = text_max_len
    
    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        encoded = [self.char2idx[self.sos_token]] + [self.char2idx[char] for char in text] + [self.char2idx[self.eos_token]]
        # Pad the sequence
        encoded += [self.char2idx[self.pad_token]] * (self.text_max_len - len(encoded))
        return encoded


    def decode(self, tokens: List[int], **kwargs: Dict[str, Any]) -> str:
        # Remove the <SOS> and <EOS> tokens
        tokens = [
            token for token in tokens 
            if token not in 
            [self.char2idx[self.sos_token], self.char2idx[self.eos_token], self.char2idx[self.pad_token]]
        ]
        return ''.join([self.idx2char.get(token, '') for token in tokens])
    

    def batch_decode(self, batch_tokens: List[List[int]], **kwargs: Dict[str, Any]) -> List[str]:
        return [self.decode(tokens) for tokens in batch_tokens]


    def __len__(self):
        return len(self.chars)