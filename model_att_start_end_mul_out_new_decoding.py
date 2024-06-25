import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(ffnn_hid_size, cls_num)
        self.linear_2 = nn.Linear(ffnn_hid_size, cls_num)
        self.linear_3 = nn.Linear(ffnn_hid_size, cls_num)
        self.linear_BNW = nn.Linear(ffnn_hid_size, cls_num)
        self.linear_EPW = nn.Linear(ffnn_hid_size, cls_num)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2_1 = self.linear_1(z)
        o2_2 = self.linear_2(z)
        o2_3 = self.linear_3(z)
        
        o2_BNW = self.linear_BNW(z)
        o2_EPW = self.linear_EPW(z)
        return o1 + o2_1, o1 + o2_2, o1 + o2_3, o1 + o2_BNW, o1 + o2_EPW

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.device = torch.device("cuda")

        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        
        self.test_dep_reg_dis = config.test_dep_reg_dis
        self.test_conv = config.test_conv

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.adj_embs = nn.Embedding(100, 20)
        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)
        
        if config.test_dep_reg_dis==0:
            conv_input_size = config.lstm_hid_size + 20 + config.dist_emb_size + config.type_emb_size# + 20
        elif config.test_dep_reg_dis==1:
            conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size
        elif config.test_dep_reg_dis==2:
            conv_input_size = config.lstm_hid_size + 20 + config.dist_emb_size
        elif config.test_dep_reg_dis==3:
            conv_input_size = config.lstm_hid_size + 20 + config.type_emb_size
        elif config.test_dep_reg_dis==4:
            conv_input_size = config.lstm_hid_size # + 20 + config.dist_emb_size + config.type_emb_size# + 20

        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = nn.Dropout(config.emb_dropout)
        
        if self.test_conv == 0:
            self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,
                                         config.conv_hid_size * len(config.dilation), config.ffnn_hid_size,
                                         config.out_dropout)
        elif self.test_conv==1:
            self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,
                                         828, config.ffnn_hid_size,
                                         config.out_dropout)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)
        self.linear_1 = nn.Linear(config.lstm_hid_size, config.conv_hid_size * len(config.dilation))
        self.dropout = nn.Dropout(config.conv_dropout)
        
        self.adj_conv = nn.Conv2d(20, 20, kernel_size=2, padding=0)
        
        self.bert_2_lstm = nn.Linear(config.bert_hid_size, config.lstm_hid_size)
        
    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, adjacency_matrix, flag=True):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        
        :param adjacency_matrix: [B, L+1, L+1'] <root>
        :param pos_tag: [B, L']
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())      
        
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())
        
        cln = self.cln(word_reps.unsqueeze(2), word_reps)
        
        adj_emb = self.adj_embs(adjacency_matrix)
        adj_emb = adj_emb.permute(0, 3, 1, 2).contiguous()
        adj_rep = self.adj_conv(adj_emb)
        adj_rep = adj_rep.permute(0, 2, 3, 1).contiguous()


        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        if self.test_conv==0:
            if self.test_dep_reg_dis==0:
                conv_inputs = torch.cat([adj_rep, dis_emb, reg_emb, cln], dim=-1)
            elif self.test_dep_reg_dis==1:
                conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
            elif self.test_dep_reg_dis==2:
                conv_inputs = torch.cat([adj_rep, dis_emb, cln], dim=-1)
            elif self.test_dep_reg_dis==3:
                conv_inputs = torch.cat([adj_rep, reg_emb, cln], dim=-1)
            elif self.test_dep_reg_dis==4:
                conv_inputs = torch.cat([cln], dim=-1)
        
        if self.test_conv==0:
            conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
            conv_outputs = self.convLayer(conv_inputs)
            conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
            outputs1, outputs2, outputs3, outputs_BNW, outputs_EPW = self.predictor(word_reps, word_reps, conv_outputs)
        elif self.test_conv==1:
            conv_inputs = torch.cat([adj_rep, dis_emb, reg_emb, cln], dim=-1)
            outputs1, outputs2, outputs3, outputs_BNW, outputs_EPW = self.predictor(word_reps, word_reps, conv_inputs)

        return outputs1, outputs2, outputs3, outputs_BNW, outputs_EPW
