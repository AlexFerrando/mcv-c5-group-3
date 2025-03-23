from torchtune.modules.tokenizers._utils import BaseTokenizer
from typing import List, Dict, Any
from transformers import AutoTokenizer

class CharacterTokenizer(BaseTokenizer):
    def __init__(
            self,
            text_max_len: int = 201
        ):
        super().__init__()
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.chars = [
            self.sos_token, self.eos_token, self.pad_token, '\n', ' ', '!', '"', '#', '%', '&', "'", '(', ')', '+', ',',
            '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '9', ':', ';', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '\x92', '\x96', '\xa0', '®', 'Á', 'É', 'à', 'á', 'â', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï',
            'ñ', 'ò', 'ó', 'ô', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ō', 'ơ', '̀', '́', '̃', '̉', 'С', 'и', 'к', 'н', 'р', 'ы',
            '\u2009', '–', '—', '‘', '’', '“', '”', '강', '개', '닭', '된', '장', '전', '정', '찌', '파'
        ]
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
            if token not in [
                self.char2idx[self.sos_token], 
                self.char2idx[self.eos_token], 
                self.char2idx[self.pad_token]
            ]
        ]
        return ''.join([self.idx2char.get(token, '') for token in tokens])
    

    def batch_decode(self, batch_tokens: List[List[int]], **kwargs: Dict[str, Any]) -> List[str]:
        return [self.decode(tokens) for tokens in batch_tokens]


    def __len__(self):
        return len(self.chars)
    
class WordTokenizer(BaseTokenizer):
    def __init__(self,text_max_len: int = 201):
        super().__init__()
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.text_max_len = text_max_len

        self.vocab = [self.sos_token, self.eos_token, self.pad_token, self.unk_token]

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_from_texts(self, texts: List[str]):
        words = set()
        for text in texts:
            tokens = text.split()
            words.update(tokens)
        self.vocab += sorted(words)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text: str) -> List[int]:
        tokens = [self.sos_token] + text.split()[:self.text_max_len - 2] + [self.eos_token]
        token_ids = [
            self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens
        ]
        token_ids += [self.word2idx[self.pad_token]] * (self.text_max_len - len(token_ids))
        return token_ids

    def decode(self, tokens: List[int]) -> str:
        return " ".join([
            self.idx2word.get(token, self.unk_token)
            for token in tokens
            if token not in {
                self.word2idx.get(self.sos_token, -1),
                self.word2idx.get(self.eos_token, -1),
                self.word2idx.get(self.pad_token, -1)
            }
        ])

    def batch_decode(self, batch_tokens: List[List[int]]) -> List[str]:
        return [self.decode(tokens) for tokens in batch_tokens]

    def __len__(self):
        return len(self.vocab)
    
class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name="bert-base-cased", text_max_len=201):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.text_max_len = text_max_len

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.text_max_len,
            truncation=True,
            padding="max_length"
        )

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def batch_decode(self, batch_tokens: List[List[int]]) -> List[str]:
        return self.tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)

    def __len__(self):
        return self.tokenizer.vocab_size

# Function to remove blank spaces in BERT Tokenizer
def clean_decoded_text(text: str) -> str:
    text = text.replace(" - ", "-")        
    text = text.replace(" ’ ", "’")         
    text = " ".join(text.split())           
    return text