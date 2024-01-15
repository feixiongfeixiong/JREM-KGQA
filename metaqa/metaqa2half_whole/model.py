import networkx
import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from allennlp.modules.attention.cosine_attention import CosineAttention



class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        # self.fc_v = nn.Linear(dim, dim)
        # self.fc_e = nn.Linear(dim, dim)

    def forward(self, v,e):

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)

        # v_output = self.fc_v(v_intermediate)
        # e_output = self.fc_e(e_intermediate)


        return v_output, e_output

class TextCNN(nn.Module):
    def __init__(self, kernel_sizes, num_channels,q_dim,q_dim1,sequence_size):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout=nn.Dropout(0.5)
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Sequential(nn.Conv1d(in_channels=q_dim,
                                                      out_channels=c,
                                                      # 这里输出通道数为[150,150],即有150个卷积核大小为3*embedding_size和4*embedding_size的卷积核
                                                      kernel_size=k),  # 卷积核的大小，这里指3和4
                                            nn.ReLU(),  # 激活函数
                                            nn.MaxPool1d(sequence_size - k + 1)))  # 池化，在卷积后的30-k+1个结果中选取一个最大值，30表示的事每条数据由30个词组成，
        self.linear1 = nn.Linear(sum(num_channels), q_dim1)  # 全连接层，输入为512维向量，输出为3维，即分类数
        self.linear2=nn.Linear(1,sequence_size)
        # self.linear3=nn.Linear(sequence_size,1)
    def forward(self, x):
        # print(x.shape)
        # embed = self.embedding(inputs)  # [30,64,300]
        # embed = embed.permute(1, 2, 0)  # [64,300,30]，这一步是交换维度，为了符合后面的卷积操作的输入
        x=x.transpose(2,1)
        # 在下一步的encoding中经过两层textcnn之后，每一层都会得到一个[64,256,1]的结果，squeeze之后为[64,256],然后将两个结果拼接得到[64,512]
        encoding = torch.cat([conv(x).squeeze(-1) for conv in self.convs], dim=1)  # [64,512]
        outputs = self.linear1(self.dropout(encoding))  # 将[64,512]输入到全连接层，最终得到[64,256]的结果
        outputs=F.relu(outputs)
        # outputs=outputs.unsqueeze(dim=2)
        # outputs=self.linear2(outputs)
        # outputs=F.relu(outputs)
        # outputs=outputs.transpose(2,1)
        # outputs=self.self_attention(outputs)
        # outputs=outputs.transpose(2,1)
        # outputs=self.linear3(outputs)
        # outputs=outputs.squeeze(dim=2)
        return outputs

class RelationExtractor(nn.Module):

    def __init__(self,r_embeddings,onehop_vage_,embedding_dim, hidden_dim, vocab_size, relation_dim, num_entities, pretrained_embeddings, device, entdrop, reldrop, scoredrop, l3_reg, model, ls, w_matrix, bn_list, freeze=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.bn_list = bn_list
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.onehop_vage_=onehop_vage_
        if self.model == 'DistMult':
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'Rotat3':
            multiplier = 3
            self.getScores = self.Rotat3
        elif self.model == 'TuckER':
            W_torch = torch.from_numpy(np.load(w_matrix))
            self.W = nn.Parameter(
                torch.Tensor(W_torch),
                requires_grad = True
            )
            # self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)),
            #                         dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('Model is', self.model)
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings1=nn.Embedding(vocab_size,embedding_dim)
        self.n_layers = 1
        self.bidirectional = True

        self.num_entities = num_entities
        self.loss = torch.nn.BCELoss(reduction='sum')

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.pretrained_embeddings = pretrained_embeddings
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        self.r_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(r_embeddings),freeze=self.freeze)
        self.onehop_vage_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(self.onehop_vage_),freeze=self.freeze)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 256
        self.mid2 = 256

        self.lin1 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)
        self.lin2 = nn.Linear(self.mid1, self.mid2, bias=False)
        xavier_normal_(self.lin1.weight.data)
        xavier_normal_(self.lin2.weight.data)
        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)
        self.hidden2rel_base = nn.Linear(hidden_dim * 2, self.relation_dim)

        self.lin11 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)
        self.lin21 = nn.Linear(self.mid1, self.mid2, bias=False)
        xavier_normal_(self.lin11.weight.data)
        xavier_normal_(self.lin21.weight.data)
        self.hidden2rel1 = nn.Linear(self.mid2, self.relation_dim)
        self.hidden2rel_base1 = nn.Linear(hidden_dim * 2, self.relation_dim)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)

        for i in range(3):
            for key, value in self.bn_list[i].items():
                self.bn_list[i][key] = torch.Tensor(value).to(device)


        self.bn0.weight.data = self.bn_list[0]['weight']
        self.bn0.bias.data = self.bn_list[0]['bias']
        self.bn0.running_mean.data = self.bn_list[0]['running_mean']
        self.bn0.running_var.data = self.bn_list[0]['running_var']

        self.bn2.weight.data = self.bn_list[2]['weight']
        self.bn2.bias.data = self.bn_list[2]['bias']
        self.bn2.running_mean.data = self.bn_list[2]['running_mean']
        self.bn2.running_var.data = self.bn_list[2]['running_var']

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)
        self.GRU1 = nn.LSTM(embedding_dim, 200, self.n_layers, bidirectional=self.bidirectional,
                           batch_first=True)
        self.GRU2 = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional,
                           batch_first=True)
        #____________________________________________________________________________________________________________________________
        self.k = 2
        # self.textcnn1=TextCNN([2,2],[32,32],400,30,3)
        # self.textcnn2=TextCNN([1,3,6],[3,8,8],3,30,400)
        # self.second_stage_linear1=nn.Linear(83,1)
        self.parameters_a = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.parameters_b = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.parameters_c = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.linear1=nn.Linear(1312,400)
        self.textcnn_pred_liear=nn.Linear(1,64)
        self.textcnn = TextCNN(kernel_sizes=[3], num_channels=[32], q_dim=800, sequence_size=64, q_dim1=400)
        self.linearonehop=nn.Linear(512,400)
        self.cosine__Attention=CosineAttention()
        self.crossCompressUnit=CrossCompressUnit(400)


    #(self, kernel_sizes, num_channels,q_dim,q_dim1,sequence_size):
    def applyNonLinear(self, outputs):
        outputs = self.lin1(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin2(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs
    def applyNonLinear2(self, outputs):
        outputs = self.lin11(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin21(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2rel1(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs
    def TuckER(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        x = head.view(-1, 1, head.size(1))

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, head.size(1), head.size(1))
        W_mat = self.rel_dropout(W_mat)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, head.size(1))
        x = self.bn2(x)
        x = self.score_dropout(x)

        x = torch.mm(x, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def RESCAL(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        ent_dim = head.size(1)
        head = head.view(-1, 1, ent_dim)
        relation = relation.view(-1, ent_dim, ent_dim)
        relation = self.rel_dropout(relation)
        x = torch.bmm(head, relation)
        x = x.view(-1, ent_dim)
        x = self.bn2(x)
        x = self.score_dropout(x)
        x = torch.mm(x, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred

    def SimplE(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = torch.mm(s, self.embedding.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx_onehop(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.onehop_vage_embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        # return pred
        return score

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        # return pred
        return score

    def Rotat3(self, head, relation):
        pi = 3.14159265358979323846
        relation = F.hardtanh(relation) * pi
        r = torch.stack(list(torch.chunk(relation, 3, dim=1)), dim=1)
        h = torch.stack(list(torch.chunk(head, 3, dim=1)), dim=1)
        h = self.bn0(h)
        h = self.ent_dropout(h)
        r = self.rel_dropout(r)

        r = r.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        x = h[0]
        y = h[1]
        z = h[2]

        # need to rotate h by r
        # r contains values in radians

        for i in range(len(r)):
            sin_r = torch.sin(r[i])
            cos_r = torch.cos(r[i])
            if i == 0:
                x_n = x.clone()
                y_n = y * cos_r - z * sin_r
                z_n = y * sin_r + z * cos_r
            elif i == 1:
                x_n = x * cos_r - y * sin_r
                y_n = x * sin_r + y * cos_r
                z_n = z.clone()
            elif i == 2:
                x_n = z * sin_r + x * cos_r
                y_n = y.clone()
                z_n = z * cos_r - x * sin_r

            x = x_n
            y = y_n
            z = z_n

        s = torch.stack([x, y, z], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = s.permute(1, 0, 2)
        s = torch.cat([s[0], s[1], s[2]], dim = 1)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    #
    # def forward(self, sentence, p_head, p_tail, question_len):
    #     embeds = self.word_embeddings(sentence)
    #     packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
    #     outputs, (hidden, cell_state) = self.GRU(packed_output)
    #     outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
    #     outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
    #     # outputs = self.drop1(outputs)
    #     # rel_embedding = self.hidden2rel(outputs)
    #     rel_embedding = self.applyNonLinear(outputs)
    #     p_head_list=p_head
    #     p_head = self.embedding(p_head)
    #     pred1 = self.getScores(p_head, rel_embedding)
    #     topk = torch.topk(pred1, k=self.k, largest=True, sorted=True)
    #     pred=self.second_stage_predict(topk, p_head_list,sentence,question_len)
    #     p_tail=p_tail.gather(1,topk[1])
    #     actual = p_tail
    #     if self.label_smoothing:
    #         actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1))
    #     loss = self.loss(pred, actual)
    #     # reg = -0.001
    #     # best: reg is 1.0
    #     # self.l3_reg = 0.002
    #     # self.gamma1 = 1
    #     # self.gamma2 = 3
    #     if not self.freeze:
    #         if self.l3_reg:
    #             norm = torch.norm(self.embedding.weight, p=3, dim=-1)
    #             loss = loss + self.l3_reg * torch.sum(norm)
    #     return loss

    def get_question_embedding(self,sentence,question_len):
        embeds = self.word_embeddings(sentence)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def get_init_embedding(self,sentence,question_len):
        embeds = self.word_embeddings(sentence)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU1(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def get_onehop_embedding(self,sentence,question_len):
        embeds = self.word_embeddings(sentence)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU2(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def get_onehop_embedding1(self,sentence,question_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def get_question_embedding1(self,sentence,question_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def get_init_embedding1(self,sentence,question_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU1(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        return outputs

    def yuanshi_predict(self,cross_relation,p_head):
        # rel_embedding=self.applyNonLinear(question_embedding)
        scores=self.getScores(self.embedding(p_head),cross_relation)
        return scores

    def fuse_predict(self,question_embedding,p_head):
        combine = torch.cat([question_embedding, self.embedding(p_head)], dim=1)
        combine = combine.unsqueeze(dim=2)
        combine = self.textcnn_pred_liear(combine)
        combine = F.relu(combine)
        combine = combine.transpose(2, 1)
        combine = self.textcnn(combine)
        # combine=self.textcnn_ans_linear(combine)
        # combine=F.relu(combine)
        re_combine, im_combine = torch.chunk(combine, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)
        pred = torch.mm(re_combine, re_tail.transpose(1, 0)) + torch.mm(im_combine, im_tail.transpose(1, 0))
        return pred

    def cross_question_relation(self,question_init_embedding):
        ###先使用注意力机制，与知识图谱中的关系交互，生成一个加权和
        relations = self.r_embedding.weight
        relations = relations.unsqueeze(dim=0)
        relations = relations.repeat(question_init_embedding.shape[0], 1, 1)
        scores = self.cosine__Attention(question_init_embedding, relations)
        generated_relation = torch.mm(scores, self.r_embedding.weight)# (batchsize,relation_dim)

        ###最大注意力
        # relations = self.r_embedding.weight
        # relations = relations.unsqueeze(dim=0)
        # relations = relations.repeat(question_init_embedding.shape[0], 1, 1)
        # scores = self.cosine__Attention(question_init_embedding, relations)
        # topk = torch.topk(scores, 5, largest=True, sorted=True)
        # topk_relations=self.r_embedding(topk[1])
        # topk_scores=topk[0].unsqueeze(dim=1)
        # generated_relation = torch.bmm(topk_scores, topk_relations)# (batchsize,relation_dim)
        # generated_relation=generated_relation.squeeze(1)
        ####
        # 通过交叉压缩矩阵交换信息
        q_output,r_output=self.crossCompressUnit(question_init_embedding,generated_relation)
        return q_output,r_output

    def forward(self,sentence, p_head, p_tail, question_len):
        # question_embedding=self.get_question_embedding(sentence,question_len)
        question_init_embedding=self.get_init_embedding(sentence,question_len)
        cross_question,cross_relation=self.cross_question_relation(question_init_embedding)
        # question_onehop_embedding=self.get_onehop_embedding(sentence,question_len)
        # #原始分数
        yuanshi_scores=self.yuanshi_predict(cross_relation,p_head)
        # #融合分数
        fuse_scores=self.fuse_predict(cross_question,p_head)
        # #onehop_scores
        onehop_scores=self.onehop_predict(cross_relation,p_head)
        # #最终分数
        final_scores=torch.sigmoid(fuse_scores*self.parameters_b+self.parameters_a*yuanshi_scores+self.parameters_c*onehop_scores)

        # topk = torch.topk(final_scores, k=self.k, largest=True, sorted=True)
        # pred=topk[0]
        # actual=p_tail.gather(1,topk[1])
        #
        pred=final_scores
        actual=p_tail
        #
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1))
        loss = self.loss(pred, actual)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss



    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)
        return rel_embedding

    def onehop_predict(self,cross_relation,p_head):
        scores = self.ComplEx_onehop(self.embedding(p_head), cross_relation)
        return scores

    def get_score_ranked(self, head, sentence, sent_len):
        question_init_embedding = self.get_init_embedding1(sentence, sent_len)
        cross_question, cross_relation = self.cross_question_relation(question_init_embedding)
        # 原始分数
        yuanshi_scores = self.yuanshi_predict(cross_relation, head.unsqueeze(0))
        # 融合分数
        fuse_scores = self.fuse_predict(cross_question, head.unsqueeze(0))
        #邻居分数
        onehop_scores=self.onehop_predict(cross_relation,head.unsqueeze(0))
        # 最终分数
        final_scores=torch.sigmoid(fuse_scores*self.parameters_b+self.parameters_a*yuanshi_scores+self.parameters_c*onehop_scores)

        topk = torch.topk(final_scores, k=self.k, largest=True, sorted=True)
        return topk





