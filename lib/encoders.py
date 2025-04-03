"""VSE modules"""
import os
import torch
import torch.nn as nn
import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel

from lib.coding import get_coding, get_pooling
from lib.modules import l2norm, SelfAttention, Transformer
from lib.Rs_GCN import Rs_GCN

import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, text_enc_type="bigru", 
                    use_bi_gru=True, no_txtnorm=False, **args):
    """A wrapper to text encoders."""
    if text_enc_type == "bigru":
        txt_enc = EncoderTextBigru(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=use_bi_gru, no_txtnorm=no_txtnorm, **args)
    elif text_enc_type == "bert":
        txt_enc = EncoderTextBert(embed_size, no_txtnorm=no_txtnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(text_enc_type))
    return txt_enc


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, **args):
    """A wrapper to image encoders."""
    img_enc = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, **args)
    return img_enc


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, split_size=16, position_embed_size=200, **args):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.position_size = split_size * split_size
        self.position_embed_size = position_embed_size
        self.position_embedding = nn.Embedding(self.position_size + 1, self.position_embed_size)
        self.fc = nn.Linear(img_dim + self.position_embed_size, embed_size)
        self.fc_att = nn.Linear(img_dim, position_embed_size) # 2048 => 200
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        if precomp_enc_type=="basic":
            self.feedforward = nn.Identity()
        elif precomp_enc_type=="selfattention":
            self.feedforward = SelfAttention(embed_size)
        elif precomp_enc_type=="transformer":
            self.feedforward = Transformer(embed_size)
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        
    def attention_forward(self, images, box_features):
        # images shape: batch_size * 36, 2048
        # box_features shape: batch_size * 36, 15, 200
        # return shape: batch_size * 36, 15

        # image_attention: batch_size * 36, 200 => batch_size * 36, 1, 200
        # box_attention: batch_size * 36, 200, 15
        image_attention = self.fc_att(images)
        image_attention = image_attention.view(image_attention.size(0), 1, image_attention.size(1))
        #print("Image attetion", image_attention)
        box_attention = torch.transpose(box_features, 1, 2)
        #print("Box attention", box_attention)
        # cos_sim: batch_size * 36, 1, 15
        # => batch_size * 36, 15
        cos_sim = torch.bmm(image_attention, box_attention)
        cos_sim = torch.tanh(cos_sim.squeeze())
        #print("Cos sim", cos_sim)
        cos_sim = F.softmax(cos_sim) # nn.Softmax()(cos_sim)
        #print("Cos sim (2)", cos_sim)
        #print("Cos sum", torch.sum(cos_sim,dim=1))
        return cos_sim

    def forward(self, images, boxes):
        # """Extract image feature vectors."""
        # features = self.fc(images)

        # features = self.feedforward(features)

        # if not self.no_imgnorm:
        #     features = l2norm(features, dim=-1)
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        #print "Images shape in forward", images
        #box_features = self.extract_box_feature(boxes) # not use
        #print("Boxes shape", boxes)
        new_boxes = boxes.view(boxes.size(0)*boxes.size(1),boxes.size(2))
        #print("New boxes shape", new_boxes)

        # new_boxes_index = new_boxes[:,:int(new_boxes.size(1)/2)].type(torch.LongTensor)
        # new_boxes_weight = new_boxes[:,int(new_boxes.size(1)/2):]
        new_boxes_index = new_boxes[:,:14].type(torch.LongTensor)
        new_boxes_weight = new_boxes[:,int(new_boxes.size(1)/2):int(new_boxes.size(1)/2+14)]
        if torch.cuda.is_available():
            new_boxes_index = new_boxes_index.cuda()
            new_boxes_weight = new_boxes_weight.cuda()
        # new_boxes_index shape: batch_size*36, 15
        # new_boxes_weight shape: batch_size*36, 15
        box_features = self.position_embedding(new_boxes_index)
        # box_features shape: batch_size*36, 15, 200
        # => batch_size*36, 200, 15

        # attention shape: batch_size*36, 15
        attention = self.attention_forward(images.view(images.size(0)*images.size(1), images.size(2)), box_features)
        new_boxes_weight = new_boxes_weight * attention
        #new_boxes_weight = l1norm(new_boxes_weight, 1)
        new_boxes_weight = F.softmax(new_boxes_weight, dtype=torch.float64) # nn.Softmax()(new_boxes_weight)
        #print("New boxes weights", new_boxes_weight)

        box_features = torch.transpose(box_features, 1, 2)
        box_features = torch.tensor(box_features, dtype=torch.float64)
        # => batch_size*36, 200, 1
        box_features = torch.bmm(box_features, new_boxes_weight.unsqueeze(2))
        # => batch_size, 36, 200
        box_features = box_features.view(boxes.size(0), boxes.size(1),-1)
        #print("Box feature shape", box_features)
        #features = self.fc(images)
        image_position = torch.cat((images, box_features),dim=2)
        image_position = torch.tensor(image_position, dtype=torch.float32)
        #print("Image postion shape", image_position)
        features = self.fc(image_position) # ##########
        #features = self.fc(images)
        #print("Final feature shape", features)
        # normalize in the joint embedding space
        GCN_img_emd = features.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        features = self.feedforward(features)
        
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


# Language Model with BiGRU
class EncoderTextBigru(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False, **args):
        super(EncoderTextBigru, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        hidden_size = embed_size
        self.rnn = nn.GRU(word_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.fc = nn.Linear(hidden_size, embed_size)
        self.init_weights(wemb_type=args["wemb_type"],word2idx=args["word2idx"],word_dim=word_dim)

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir+wemb_type)
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            ##
            self.embed.requires_grad = False
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        # x_emb:(128, 30, 300)
        x_emb = self.embed(x)
        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)
        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb:(128, 30, 2048), cap_len:(128)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


# Language Model with BERT
class EncoderTextBert(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        root = os.path.expanduser("/home/ubuntu/Students/zhoutao/data/FNE_dataset/google/bert-base-uncased")
        self.bert = BertModel.from_pretrained(config=root,pretrained_model_name_or_path=root)
        
        self.linear = nn.Linear(768, embed_size)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb

class SimsEncoder(nn.Module):
    def __init__(self, coding_type, pooling_type, **args):
        super(SimsEncoder, self).__init__()
        self.opt = args["opt"]
        self.coding = get_coding(coding_type, opt=self.opt)
        self.pooling = get_pooling(pooling_type, opt=self.opt)
    # img_emb:(128, 36, 1024), (128 ,30, 1024), img_lens(128), cap_lens:(128)
    def forward(self, img_emb, cap_emb, img_lens, cap_lens):
        # sims:(128, 128, 36)
        sims = self.coding(img_emb, cap_emb, img_lens, cap_lens)
        sims = self.pooling(sims)
        return sims
    

    