import torch 
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self,
                 hidden_size: int = 512, 
                 num_heads: int = 12, 
                 max_len: int = 1024, # same as vocab size
                 dim_ff: int = 2048,
                 out_size: int = 1024,
                 in_size: int = 1024,
                 word_embedding_dim: int = 128,
                 device: str = 'cuda',
                 dim_q: int = 128,
                 dim_k: int = 128,
                 dim_v: int = 128,
                 blocks: int = 12
                 ):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = word_embedding_dim
        self.max_len = max_len
        self.dim_ff = dim_ff
        self.out_size = out_size
        self.in_size = in_size
        self.device = device
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        
        # embeddings 
        self.word_embedding = self.create_positional_embedding(self.max_len, self.word_embedding_dim)
        self.pos_embedding = nn.Embedding(self.max_len, self.word_embedding_dim)
        self.time_embedding = nn.Embedding(self.max_len, self.word_embedding_dim)
        
        # attention blocks 
        encoder_attn_blocks = []
        for _ in range(blocks):
            block = Block(embedding_size = self.hidden_size,
                          num_heads=12,
                          dim_q=self.dim_q, 
                          dim_k=self.dim_k,
                          dim_v=self.dim_v,
                          )
            encoder_attn_blocks.append(block)
            
        decoder_attn_blocks = []
        for _ in range(blocks):
            block = Block(embedding_size = self.hidden_size,
                          num_heads=12,
                          dim_q=self.dim_q, 
                          dim_k=self.dim_k,
                          dim_v=self.dim_v,
                          )
            decoder_attn_blocks.append(block)
            
        self.encoder_attn_blocks = nn.Sequential(*encoder_attn_blocks)
        self.decoder_attn_blocks = nn.Sequential(*decoder_attn_blocks)
        
        self.final_ff = nn.Linear(self.hidden_size, self.out_size)
    
    def create_positional_embedding(self, max_len, embed_dim):
        position = np.arange(max_len)[:, np.newaxis]  # Shape (max_len, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))  # Shape (embed_dim/2,)
        
        pos_embedding = np.zeros((max_len, embed_dim))
        pos_embedding[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
        pos_embedding[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
        
        return torch.tensor(pos_embedding, dtype=torch.float32)
    
    def encode(self, x):
        source_embedding = self.word_embedding(x)
        source_embedding = source_embedding + self.pos_embedding(x)
        source_embedding = source_embedding + self.time_embedding(x)
        source_embedding = self.encoder_attn_blocks(source_embedding)
        return source_embedding
    
    def decode(self, x, encodings):
        target_embedding = self.word_embedding(x)
        target_embedding = target_embedding + self.pos_embedding(x)
        target_embedding = target_embedding + self.time_embedding(x)
        for block in self.decoder_attn_blocks:
            x = block(target_embedding, encodings)
        return x
    
    def forward(self, x):
        encodings = self.encode(x)
        x = self.decode(x, encodings)
        x = self.final_ff(x)
        x = nn.Softmax(dim=-1)(x)
        return x


class Block(nn.Module):
    def __init__(self, embedding_size: int = 128, 
                 num_heads: int = 12, 
                 dim_q: int = 128, 
                 dim_k: int = 128, 
                 dim_v: int = 128, 
                 dropout: float = 0.1):
        super(Block, self).__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dropout = dropout
        
        # self.qkv = nn.Linear(hidden_size, dim_q + dim_k + dim_v)
        self.dropout = nn.Dropout(dropout)
        self.attn_proj = nn.Linear(num_heads * dim_v, embedding_size)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(embedding_size)
        self.softmax = nn.Softmax(dim=-1)
        
        self.qk = nn.Linear(embedding_size, self.num_heads * (self.dim_q + self.dim_k))
        self.v = nn.Linear(embedding_size, self.num_heads * self.dim_v)
        self.ff = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x, encodings=None):
        batch_size, seq_len, _ = x.shape
        qk = self.qk(x)
        
        if encodings is not None:
            v = self.v(encodings)
        else:
            v = self.v(x)
            
        qk = qk.reshape(batch_size, seq_len, 2, self.num_heads, -1).permute(2, 0, 1, 3, 4) # (q or k or v, batch, seq len, num heads, embedding)
        q, k = qk.chunk(2, dim=0)
        v = v.reshape(batch_size, seq_len, self.num_heads, -1)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        kT = k.transpose(-2, -1)
        attn = q @ kT / np.sqrt(self.dim_k)
        attn = attn.softmax(-1)

        x_ = attn @ v # now is shape (batch, seq len, num heads, embedding)
        x_ = self.dropout(x_)
        x_ = x_.view(x_.shape[0], x_.shape[1], -1)
        x_ = self.attn_proj(x_)
        out = self.ln(x_ + x)
        
        out = out.reshape(batch_size, seq_len, -1)
        out = self.ff(out)
        out = self.activation(out)
        out = self.ln(out + x)
        return out

if __name__ == '__main__':
    t = Transformer()
    x = torch.randint(0, 12, (1, 10))
    print(t(x).shape)