from numpy.testing._private.utils import assert_no_gc_cycles
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import AutoModel
import random
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer
import matplotlib.ticker as ticker
from sklearn.metrics.pairwise import cosine_similarity
import arabic_reshaper
from bidi import algorithm as bidialg
from bidi.algorithm import get_display
from mpl_toolkits.axes_grid1 import make_axes_locatable

class KGEEnrichment(nn.Module):
   def __init__(self, model_name, embedding_matrix, idx2entity):
     super().__init__()

     self.idx2entity= idx2entity
     # define layers and parameters 
     self.model_name = model_name
     config = BertConfig.from_pretrained(self.model_name , output_hidden_states=True)
     self.bert_model = BertModel.from_pretrained(self.model_name, config=config)
     
     for param in self.bert_model.parameters():
            param.requires_grad = True
     
     self.embedding_matrix_tensor = torch.stack(embedding_matrix)
     self.embeddings = nn.Embedding.from_pretrained(self.embedding_matrix_tensor, freeze= True)
     self.concept_vocab_size = self.embedding_matrix_tensor.size()[0]
     self.concept_dim = self.embedding_matrix_tensor.size()[1]
     self.bert_encoder_dim= 768
     self.a1 = nn.Linear(self.bert_encoder_dim,self.concept_dim, bias=False)
     #self.output_layer= nn.Linear(self.bert_encoder_dim+self.concept_dim,2 )
     self.output_layer= nn.Linear(self.bert_encoder_dim,2 )
     self.concat_output_layer= nn.Linear(self.bert_encoder_dim+ 128 ,2 )
     
     # for gating
     self.g_lin1 = nn.Linear(self.bert_encoder_dim, self.bert_encoder_dim)
     self.g_lin2 = nn.Linear(self.bert_encoder_dim, self.bert_encoder_dim)
     self.projecting_mem = nn.Linear(self.concept_dim, self.bert_encoder_dim, bias=False )
     
     # for self-matching
     #self.drop_prob = drop_prob
     self.weight1 = nn.Parameter(torch.zeros(self.bert_encoder_dim, 1))
     self.weight2 = nn.Parameter(torch.zeros(self.bert_encoder_dim, 1))
     self.weight_mul = nn.Parameter(torch.zeros(1, 1, self.bert_encoder_dim))
     for weight in (self.weight1, self.weight2, self.weight_mul):
            nn.init.xavier_uniform_(weight)
     self.bias = nn.Parameter(torch.zeros(1))
     self.tokenizer = self.load_tokenizer()
     
 

   def load_tokenizer(self):
        
        return AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English")
   
   def dynamic_expand(self, dynamic_tensor, smaller_tensor):
    """
    :param dynamic_tensor:
    :param smaller_tensor:
    :return:
    """
    scale=0.0
    assert len(dynamic_tensor.shape) > len(smaller_tensor.shape)
    if type(smaller_tensor.shape) == list:
        for dim_idx, dim in smaller_tensor.shape:
            dynamic_tensor_dim_idx = len(dynamic_tensor) - len(smaller_tensor) + dim_idx
            assert dynamic_tensor.shape[dynamic_tensor_dim_idx] % dim == 0
    elif type(smaller_tensor.shape) == int:
        assert dynamic_tensor.shape[-1] % smaller_tensor.shape == 0
    memory_embs_zero = dynamic_tensor* scale
    smaller_tensor = torch.add(memory_embs_zero, smaller_tensor)
    return smaller_tensor

   def get_similarity_matrix(self, g_out): # [batch_size, seq_len, bert_dim]
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        g_len = g_out.size(1)
        # TODO: check dropout
        #c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        #q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(g_out, self.weight1).expand([-1, -1, g_len])
        #print("s0 shape ", s0.shape)
        s1 = torch.matmul(g_out, self.weight2).transpose(1, 2)\
                                           .expand([-1, g_len, -1])
        #print("s1 shape ", s1.shape)
        s2 = torch.matmul(g_out * self.weight_mul, g_out.transpose(1, 2))
        #print("s2 shape ", s2.shape)

        s = s0 + s1 + s2 + self.bias
        #print("s shape ", s.shape)

        return s
  
   def masked_softmax(self, logits, mask, dim=-1):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    probs = F.softmax(masked_logits, dim)

    return probs

   def forward(self, input_ids, attention_mask, token_type_ids, start_positions, end_positions, concept_ids, all_padded_concepts_mask, i_batch, is_training, analysis):
     
     
     # Layer 1: BERT layer
     # [0] at the end to return last hidden layer representation 
     
     output= self.bert_model(input_ids, attention_mask= attention_mask,token_type_ids= token_type_ids )
     bert_output = output[0]
     #print("bert shape :", bert_output.shape) #[batch_size, max_Seq, bert_dim ]
     
     #Layer 2: Attention layer
     bert_input_dim = bert_output_dim = bert_output.size(2)
     #print("concept size before embedding ", concept_ids.shape) #[batch_size, seq_len, max_concepts_len]
    
     # retrieve concepts embedding 
     # get memory embedding 
     # concept_ids [batch_size, seq_len, max_conepts_len]
     
     concepts_embeddings = self.embeddings(concept_ids) #[batch_size, seq_len, max_conepts_len, emb_size]
     #print("concepts embedding : ", concepts_embeddings)
     concepts_dim = concepts_embeddings.size(3) #[batch_size, seq_len, max_conepts_len, emb_size]
     
     batch_size = bert_output.size(0)
     seq_len = bert_output.size(1)   # bert_vector seq_length 
     max_concepts_len = concepts_embeddings.size(2) # max_concepts_length 

       
     projected_bert = self.a1(bert_output)# [batch_size, seq_len,c_emb_size ]
     expanded_bert = torch.unsqueeze(projected_bert, 2) #[batch_size, seq_len, 1, c_emb_size ]
     #print("bert expanded shape: ", expanded_bert.shape)
     #print("concepts shape: ", concepts_embeddings.shape)

     expanded_concepts_emb= concepts_embeddings.permute(0,1,3,2)
     #print("expanded_concepts_emb shape: ", expanded_concepts_emb.shape)

     memory_score = torch.matmul(expanded_bert, expanded_concepts_emb) #[batch_size, seq_len,1, concepts_size]
     memory_score = torch.squeeze(memory_score, 2) #[batch_size, seq_len, concepts_size]
     #print("memeory_score shape ", memory_score.shape) 


     # apply masking on padded tokens
     # replace the score of padded concepts to a very small value so attention whill not pay attention to padded concepts 
     memory_score.masked_fill(all_padded_concepts_mask==0, value= -1e10)

     # Memory attention
     mem_att= F.softmax(memory_score,dim=2)
     
     
     # analysis case study 
     if analysis== True:
        seq_tokens =[] 

        # get sequence tokens
        seq= input_ids[0]
        ts= seq.cpu().detach().numpy()
        seq_t= ts.tolist()
        
        for t in seq_t:
           seq_tokens.append(self.tokenizer.convert_ids_to_tokens(t))
        print(seq_tokens)
      
        # plot attention
        ex=  memory_score #[ batch_size, seq_len, concepts_size]
        attentions= ex.cpu().detach().numpy()
        print(attentions)
        #self.showAttention(seq_tokens, attentions )
        
        
        for ex in mem_att:
            att = ex.cpu().detach().numpy()
            plt.matshow(att)
            cb = plt.colorbar()
        
     
     mem_att = torch.unsqueeze(mem_att, 2) #[batch_size, seq_len,1,  concepts_size]
     
     
     #print("mem_att: ",mem_att.shape )
     #print("memory embedding: ", concepts_embeddings.shape)# [batch_size, seq_len, concept_size, emb_dim]
     expanded_mem= torch.matmul(mem_att, concepts_embeddings)# [batch_size, seq_len, 1,emb_dim ]
     
     expanded_mem = torch.squeeze(expanded_mem, 2)
     #print("expanded memory: ", expanded_mem.shape)# [batch_size, seq_len,emb_dim ]
    
     
     # we will check two approaches
     #A) concat expanded memory with bert then output layer
     output= torch.cat((bert_output,expanded_mem), 2) 
     #print("output: ", output.shape)#[batch_size, seq_len, bert_size+emb_dim]
     
     '''
     #B) gating layer
     # gating layer
     projected_mem= self.projecting_mem(expanded_mem)#[batch_size, seq_len, bert_dim]
     #print("projected mem ",projected_mem.shape )

     g = torch.sigmoid(self.g_lin1(bert_output))
     #print("g shape ", g.shape)

     v = torch.tanh(self.g_lin2(projected_mem))
     #print("v shape ", v.shape)
     g_output = torch.mul(g,v)+ torch.mul(1-g, bert_output )# [batch_size, seq_len, bert_dim]
     #print('ouput shape ', output.shape)
     
     #self_attention (optinal)
     batch_size = g_output.size(0) 
     seq_len = g_output.size(1) 
     s = self.get_similarity_matrix(g_output)# (batch_size, seq_len, seq_len)
     #print(s)
     # apply masking on padded tokens
     # replace the score of padded concepts to a very small value so attention whill not pay attention to padded concepts
    # print("att mask ", attention_mask.shape) # [batch_size, seq_len]
     s_trans = s.permute(2,0,1)
    # print("s_trans ",s_trans.shape )
     
     softmax_mask = self.dynamic_expand(s_trans, attention_mask)  # [sq(1),bs,sq]
    # print("after dynamic expand ",softmax_mask.shape)

     softmax_mask = softmax_mask.permute(1,0,2)    # [BS,sq(1),SQ]
    # print("after transpose ",softmax_mask.shape)

     s.masked_fill(softmax_mask==0, value= -1e10)
    # print("similarty score ", s.shape)

     attn_prob = F.softmax(s, dim=2)  # [BS,SQ,SQ]
     
     for ex in attn_prob:
         plt.matshow(ex.cpu().detach().numpy())
     
     self_out = torch.matmul(attn_prob, g_output)
     #print("self_out shape ", self_out.shape)
     '''
     if analysis == True:
       print('ouput shape ', bert_output.shape)
       # I need to decide on the index
       context_o = bert_output[:,17:50,:]
       question_o = bert_output [:,:9, :]

       context_o = torch.squeeze(context_o, 0)
       question_o = torch.squeeze(question_o, 0)

       print(context_o.shape)
       print(question_o.shape)
       
       q_np= question_o.cpu().detach().numpy()
       c_np= context_o.cpu().detach().numpy()
       
       result= cosine_similarity(q_np,c_np )
       
       tokens =  self.tokenizer.convert_ids_to_tokens(input_ids[0])

       question= tokens[:9]
       context= tokens[17:50]

       fig, ax = plt.subplots()
       im= ax.matshow(result)

       # shape Arabic context tokens
       
       ar_c=[]
       for t in context:
           reshaped_text = arabic_reshaper.reshape(t)
           artext = get_display(reshaped_text)
           ar_c.append(artext)
       
       # shape Arabic context tokens
       ar_q=[]
       
       for t in question:
           reshaped_text = arabic_reshaper.reshape(t)
           artext = get_display(reshaped_text)
           ar_q.append(artext)
       
      
       # Show all ticks and label them with the respective list entries
       plt.xticks(range(len(ar_c)), ar_c, rotation='vertical')
       plt.yticks(range(len(ar_q)), ar_q)
     
       # Rotate the tick labels and set their alignment.
       #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
       #rotation_mode="anchor")
       # create an axes on the right side of ax. The width of cax will be 5%
       # of ax and the padding between cax and ax will be fixed at 0.05 inch.
       divider = make_axes_locatable(ax)
       cax = divider.append_axes("right", size="5%", pad=0.05)

       plt.colorbar(im, cax=cax)
       plt.xlabel('sequence')
       plt.ylabel('score')
       plt.show()
       
     def compute_loss(logits, positions):
            loss = F.cross_entropy(logits, positions)
            loss = torch.mean(x=loss)
            return loss

     #(option 1) softmax linear layer (output layer) after gating
     #logits = self.output_layer(g_output)

     #(option 2)self_attention output
     #logits = self.output_layer(self_out)

     #(option 3)basline model
     #logits = self.output_layer(bert_output)

     # (option 4) concat attention with bert with no gating
     logits = self.concat_output_layer(output)

     start_logits, end_logits = torch.unbind(logits, dim= 2)
     #print("start_logits: ", start_logits.shape)
     #print("end_logits: ", end_logits.shape)
     
     if is_training:
         # return loss 
         start_loss = compute_loss(start_logits, start_positions)
         end_loss = compute_loss(end_logits, end_positions)
         total_loss = (start_loss + end_loss) / 2.0

         return total_loss
     else:
         # return loss and prediction
         start_loss = compute_loss(start_logits, start_positions)
         end_loss = compute_loss(end_logits, end_positions)
         total_loss = (start_loss + end_loss) / 2.0

         return start_logits, end_logits, total_loss
     '''
    def showAttention(self, input_sentence, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + input_sentence +['<EOS>'])

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
     '''
    
