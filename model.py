import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dmodel: int):
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dmodel)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel) #[batch_size, seq_len, dmodel]

class PositionalEncoding(nn.Module):
    def __init__(self, dmodel: int, seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.dmodel = dmodel
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of size [seq_len, dmodel]
        pe = torch.zeros(seq_len, dmodel)

        #create a vector of shape [seq_len, 1]
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0)/dmodel))

        ##apply the sin, cos function to the position

        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)
        pe = pe.unsqueeze(0) #[1, seq_len, dmodel]

        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,  eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        x = (x-mean)/std+self.eps
        return self.alpha*x + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_ff, dmodel, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(dmodel, d_ff)
        self.fc_2 = nn.Linear(d_ff, dmodel)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #[batch, seq_len, dmodel] --> [batch, seq_len, d_ff] --> [batch, seq_len, dmodel]
        return self.fc_2(self.dropout(torch.relu(self.fc_1(x))))

class MultiheadAttention(nn.Module):
    def __init__(self, dmodel, n_heads, dropout):
        super().__init__()
        assert dmodel % n_heads == 0, "dmodel is not divisible by n_heads"
        self.dmodel = dmodel
        self.n_heads = n_heads
        self.d_head = self.dmodel // self.n_heads
        self.fc_q = nn.Linear(dmodel, dmodel)
        self.fc_k = nn.Linear(dmodel, dmodel)
        self.fc_v = nn.Linear(dmodel, dmodel)
        self.fc_o = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
    
        
    def forward(self, q, k, v, mask = None):
        # q, k, v= [batch_size, seq_len, dmodel]
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)
        batch_size = Q.shape[0]

        # Q, K, V =  [batch_size, seq_len, dmodel] --> [batch_size, n_heads, seq_len, d_head]

        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2) 
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head) #energy = [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e-9)
        
        attention = torch.softmax(energy, dim = - 1)

        # attention = [batch_size, n_heads, seq_len, seq_len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch_size, n_heads, seq_len, d_head]

        x = x.transpose(1, 2).contiguous() # --> contiguous so Pytorch can view it in place in the memory

        # x  = [batch_size, q_len, n_heads, d_head]

        x = x.view(batch_size, -1, self.dmodel)

        # x = [batch_size, q_len, dmodel]

        x = self.fc_o(x)
        return x, attention

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_outputs, src_mask, tgt_mask): #src mask: source language, tgt_mask: target mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_outputs, encoder_outputs, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, dmodel: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(dmodel, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, dmodel) --> (batch, seq_len, vocab_size)
        x = torch.log_softmax(self.fc(x), dim = -1)
        return x 

class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size:int, src_seq_len: int, tgt_seq_len:int, dmodel: int = 512, N: int = 6, h:int = 8, dropout: float = 0.1, d_ff: int = 2048):
    #Create the embedding layer
    src_embed = InputEmbedding(src_vocab_size, dmodel)
    tgt_embed = InputEmbedding(tgt_vocab_size, dmodel)

    #create the positional encoding
    src_pos = PositionalEncoding(dmodel, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(dmodel, tgt_seq_len, dropout)


    #Create encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(dmodel, h, dropout)
        encoder_feed_forward_block = FeedForward(d_ff, dmodel, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    #Create decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(dmodel, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(dmodel, h, dropout)
        decoder_feed_forward_block = FeedForward(d_ff, dmodel, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    #Create projection layer
    projection_layer = ProjectionLayer(dmodel, tgt_vocab_size)

    #Create the transformer
    transformer = Transformers(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
