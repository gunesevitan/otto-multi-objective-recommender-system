import torch
import torch.nn as nn


class CollaborativeFiltering(nn.Module):

    def __init__(self, n_embeddings, n_factors, sparse=False, dropout_probability=0):

        super(CollaborativeFiltering, self).__init__()

        self.embeddings = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=n_factors, sparse=sparse)
        self.bias = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=1, sparse=sparse)
        self.dropout = nn.Dropout(p=dropout_probability) if dropout_probability > 0 else nn.Identity()

    def forward(self, x1, x2):

        bias = self.bias(x1) + self.bias(x2)
        x1 = self.embeddings(x1)
        x2 = self.embeddings(x2)
        outputs = (self.dropout(x1) * self.dropout(x2)).sum(dim=1) + bias

        return outputs


class MatrixFactorization(nn.Module):

    def __init__(self, n_sessions, n_aids, embedding_dim, sparse=False, dropout_probability=0):

        super(MatrixFactorization, self).__init__()

        self.session_embeddings = nn.Embedding(num_embeddings=n_sessions, embedding_dim=embedding_dim, sparse=sparse)
        self.aid_embeddings = nn.Embedding(num_embeddings=n_aids, embedding_dim=embedding_dim, sparse=sparse)
        self.session_bias = nn.Embedding(num_embeddings=n_sessions, embedding_dim=1, sparse=sparse)
        self.aid_bias = nn.Embedding(num_embeddings=n_aids, embedding_dim=1, sparse=sparse)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, sessions, aids):

        sessions_weight = torch.squeeze(self.session_embeddings(sessions), dim=1)
        aids_weight = torch.squeeze(self.aid_embeddings(aids), dim=1)
        bias = torch.squeeze(self.session_bias(sessions) + self.aid_bias(aids), dim=1)
        outputs = (self.dropout(sessions_weight) * self.dropout(aids_weight)).sum(dim=1, keepdim=True) + bias

        return outputs.squeeze(dim=1)
