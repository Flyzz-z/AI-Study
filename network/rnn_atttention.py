'''
主要学习rnn+attention,seq2seq模型，encoder+decoder模型
并非完整代码，仅含主要流程
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """
    QKV Attention mechanism for Seq2Seq models
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # QKV 
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)   
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
      
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Compute attention weights and context vector
        
        Args:
            encoder_outputs: [batch_size, seq_len, hidden_size]
            decoder_hidden: [batch_size, hidden_size]
            
        Returns:
            context: [batch_size, hidden_size]
            attention_weights: [batch_size, seq_len]
        """
        # Compute Q, K, V
        Q = self.W_q(decoder_hidden).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 输入是编码器所有时间步的输出
        K = self.W_k(encoder_outputs)              # [batch_size, seq_len, hidden_size]
        V = self.W_v(encoder_outputs)              # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_size)  # [batch_size, 1, seq_len]
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights, V).squeeze(1)  # [batch_size, hidden_size]
        
        return context, attention_weights.squeeze(1)
    
# Encoder正常返回所有时间步的输出，不需要注意力改造

# Decoder部分需要改造，在每个时间步计算注意力
class Decoder(nn.Module):
    """
    Decoder component of the Seq2Seq model using GRU with Attention
    Generates output sequence one token at a time
    """
    def __init__(self, vocab_size, embed_size=512, hidden_size=1024, num_layers=2, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.attention = Attention(hidden_size)

        # GRU layer for generating sequences (input: embedding + context)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=False)
        
        # Output projection layer to vocabulary (input: GRU output)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        Forward pass of the decoder with attention (single step)
        TODO: the logic of this function is inconsistent with the forward_seq function, need to fix it
        
        Args:
            input_token: Current input token [batch_size, 1]
            hidden: Hidden state from encoder/previous step [num_layers, batch_size, hidden_size]
            encoder_outputs: All encoder outputs [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Vocabulary predictions [batch_size, vocab_size]
            hidden: Updated hidden state [num_layers, batch_size, hidden_size]
            attention_weights: Attention weights [batch_size, seq_len]
        """ 
        embedded = self.embedding(input_token)  # [batch_size, 1, embed_size] 

        decoder_hidden = hidden[-1]  # [batch_size, hidden_size]

        context, attention_weights = self.attention(encoder_outputs, decoder_hidden)  # context: [batch_size, hidden_size]

        # 拼接输入和上下文
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [batch_size, 1, embed_size + hidden_size]

        gru_out, hidden = self.rnn(rnn_input, hidden)  # gru_out: [batch_size, 1, hidden_size]

        output = self.output_projection(gru_out.squeeze(1))  # [batch_size, vocab_size]

        return output, hidden, attention_weights