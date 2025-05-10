import torch
import torch.nn as nn

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=2, state_type='concat'):
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        # 自定义返回state类型（concat、add）
        self.state_type = state_type

    def forward(self, token_seq):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        if self.state_type == 'concat':
            # 最后一个时间步的隐藏状态
            hidden = outputs[:,-1,:]
        elif self.state_type == 'add':
            # 返回最后一个时间步的隐藏状态(相加)
            hidden = torch.sum(hidden, dim=0)
            outputs = outputs[...,:250] + outputs[...,250:]
        else:
            raise ValueError("state_type must be 'concat' or 'add'")
        return hidden, outputs


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output, dec_output):
        # a_t = h_t @ h_s  
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 1.计算 结合解码token和编码token，关联的权重
        a_t = torch.softmax(a_t, dim=1)
        # 2.计算 关联权重和编码token 贡献值
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout=0.5, state_type='concat'):
        super(Decoder, self).__init__()
        if state_type == 'concat':
            hidden_dim = hidden_dim * 2
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim, input_dim)  # 解码词典中词汇概率
        # attention层
        self.atteniton = Attention()
        # attention结果转换线性层
        self.atteniton_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_seq, hidden_state, enc_output):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [1, batch_size, hidden_dim]
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))

        # attention运算
        c_t = self.atteniton(enc_output, dec_output)
        # [attention, dec_output]
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        # 线性运算
        out = torch.tanh(self.atteniton_fc(cat_output))
        # dropout
        out = self.dropout(out)
        # out: [batch_size, seq_len, hidden_dim * 2]
        logits = self.fc(out)
        return logits, hidden

class Seq2Seq(nn.Module):

    def __init__(self,
                 enc_emb_size, 
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5,
                 state_type='concat'
                 ):
        
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size,state_type=state_type)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout,state_type=state_type)


    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state, outputs = self.encoder(enc_input)
        output,hidden = self.decoder(dec_input, encoder_state, outputs)

        return output,hidden

if __name__ == '__main__':
    
    # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 15
    seq_len = 10

    # encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # hidden_state = encoder(token_seq)  # Encoder输出（最后时间步状态）
    # # print(hidden_state.shape)  # 应该是 [batch_size, hidden_dim]

    # # 测试Decoder
    # decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # logits = decoder(token_seq, hidden_state)  # Decoder输出
    # print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )

    logits,_ = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

