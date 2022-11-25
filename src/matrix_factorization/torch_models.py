import torch.nn as nn


class MatrixFactorization(nn.Module):

    def __init__(self, n_aids, embedding_dim):

        super(MatrixFactorization, self).__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=n_aids,
            embedding_dim=embedding_dim,
            sparse=True
        )

    def forward(self, aid1, aid2):

        aid1 = self.embeddings(aid1)
        aid2 = self.embeddings(aid2)

        return (aid1 * aid2).sum(dim=1)
