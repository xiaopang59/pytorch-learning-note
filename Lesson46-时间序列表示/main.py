import torch
import torch
from torch import nn

word_to_ix = {"hello": 0, "world": 1}

lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)

embeds = nn.Embedding(2, 5)     # 2 words in vocab, 5 dimensional embeddings
hello_embed = embeds(lookup_tensor)
print(hello_embed)



import torchnlp

from torchnlp import word_to_vector


def main():

    # vec = word_to_vector.GloVe()
    vec = word_to_vector.BPEmb()


if __name__ == "__main__":
    main()