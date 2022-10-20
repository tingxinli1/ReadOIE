import re
from spacy.tokens import Doc
class WhitespaceTokenizer:
    """customized spacy tokenizer"""
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = [token for token in re.split(r'\s*', text) if token!='']
        return Doc(self.vocab, words=words)

def sort_span_pair(span_pair):
    """put np span appears earlier in the sentence in position 0"""
    np1, np2 = span_pair
    if np1[0] <= np2[0]:
        return [np1, np2]
    else:
        return [np2, np1]