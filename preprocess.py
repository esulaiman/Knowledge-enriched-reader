
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import numpy as np
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi import algorithm as bidialg
from bidi.algorithm import get_display


class Processing(object):

  max_length = 384 # The maximum length of a feature (question and context)
  doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
        
  
  def __init__(self, entity2idx, model_type ):
      
      self.model_type = model_type
      self.tokenizer = self.load_tokenizer()
      self.entity2idx = entity2idx
      

  def load_tokenizer(self):
        
        return AutoTokenizer.from_pretrained(self.model_type)
  
   #examples is one example
   # this will tokanize one example at a time and match it to concept to return a feature or list of features if <max_seq_len
  def prepare_train_features(self, examples, concept):

        max_length = 384 # The maximum length of a feature (question and context)
        doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
        

        pad_on_right = self.tokenizer.padding_side == "right"
        
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        #
        #
        tokenized_examples = self.tokenizer(
            examples["question"] if pad_on_right else "context",
            examples["context" ] if pad_on_right else "question",
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        # embedding for all features
        c_ids = []
        lens=[]
        for i, offsets in enumerate(offset_mapping):
       
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["text"]) == 0 or len(str(answers['answer_start']))==0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while token_start_index < len(sequence_ids)-1 and sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
            

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while token_end_index != 0  and sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
            
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index-1)
               
                    while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                       token_end_index -= 1
                    '''
                    if offsets[token_end_index +1 ][1]== end_char:
                        tokenized_examples["end_positions"].append(token_end_index+ 2)
                    else:
                    '''
                    tokenized_examples["end_positions"].append(token_end_index+ 1)
        
       

            max_concepts_len= 40
            #create conceptsE sequence
            concepts_ids= [[0]* max_concepts_len ]*len(sequence_ids)
            lengths= [0]* len(sequence_ids)
            
            # map documents concepts
            '''
            subs=[]
            from_index = 0
            if len(concept['doc_entities'])>max_concepts_len:
                concept['doc_entities']= concept['doc_entities'][:max_concepts_len]

            for e in concept['doc_entities']:
              
               if e['entity'].find('_')!= -1 and len(subs)==0:
                   subs= e['entity'].split('_')

               # one token 
               if  e['entity'].find('_')==-1:
                 try:
                    start_char = examples['context'].index(e['entity'], from_index)
                
                 except ValueError: 
                    print("couldn't find concept ", e['entity'])
                    continue
                 end_char= start_char + len(e['entity'])
                 from_index= end_char
                 
               else:

                 # correct start and end position 
                 try: 
                     start_char = examples['context'].index(subs[0], from_index)
                 except ValueError: 
                     print("couldn't find concept ", subs[0] )
                     continue
                 end_char= start_char + len(subs[0])
                 from_index= end_char
                 if len(subs)!=0:
                   #delete item in index 0 
                   subs.pop(0)

         
               #print(examples['context'][start_char:end_char])
               # Start token index of the current span in the text.
               token_start_index = 0
               while token_start_index < len(sequence_ids)-1 and sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                   token_start_index += 1
             

               # End token index of the current span in the text.
               token_end_index = len(input_ids) - 1
               while token_end_index != 0  and sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
             

               # Detect if the concept token is out of the span (do nothing).
               if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                  
                  continue
               else:
                  # Otherwise move the token_start_index and token_end_index to the two ends of the token.
                  # Note: we could go after the last offset if the answer is the last word (edge case).
                  while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                      token_start_index += 1
                  token_start_index = token_start_index-1
               
                  while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                      token_end_index -= 1
                  if offsets[token_end_index +1 ][1]== end_char:
                      token_end_index = token_end_index+ 2
                  else:
                      token_end_index = token_end_index+ 1
                  # if start and end are on the same token
                  if token_start_index== token_end_index:
                     token_end_index+=1

                  # retrieve list of concept id for this token 
                  list_con_ids = [0]* max_concepts_len
                  cons= e['concepts']
              
                  not_f=0
                  cnt=0
                  for c in cons:                 
                     try:
                       if cnt < max_concepts_len:
                           list_con_ids[cnt] = self.entity2idx[c]
                           cnt+=1
                           continue
                       else:
                         break
                     except KeyError:
                       not_f+=1
                  

                  #assign concepts ids to this token 
                  while token_start_index< token_end_index:
                    concepts_ids[token_start_index]=list_con_ids
                    lengths[token_start_index] = len(list_con_ids)
                    token_start_index+=1

             
            '''  
                  # map q concepts
                  # TODO: we must check sequence_id
            subs=[]
            from_index = 0
            
            if len(concept['question_entities'])>max_concepts_len:
                concept['question_entities']= concept['question_entities'][:max_concepts_len]

            for  e in concept['question_entities']:

               if e['entity'].find('_')!= -1 and len(subs)==0:
                   subs= e['entity'].split('_')

               # one token 
               if e['entity'].find('_')==-1:
                 try:
                    start_char = examples['question'].index(e['entity'], from_index)
                 except ValueError: 
                     print("couldn't find q concept ", e['entity'])
                     continue
                 end_char= start_char + len(e['entity'])
                 from_index= end_char
               else:

                # correct start and end position 
                 try:
                     start_char = examples['question'].index(subs[0], from_index)
                 except ValueError: 
                      print("couldn't find q concept ", subs[0] )
                      continue
                 end_char= start_char + len(subs[0])
                 from_index= end_char
                 if len(subs)!=0:
                    #delete item in index 0 
                    subs.pop(0)

               #print(examples['question'][start_char:end_char])
               # Start token index of the current span in the text.
               token_start_index = 0
           

               # End token index of the current span in the text.
               token_end_index = 1
               while token_end_index < len(sequence_ids)-1 and sequence_ids[token_end_index] ==0 :
                 token_end_index+=1

               token_end_index-=1  
             

               # Detect if the concept token is out of the span (do nothing).
               if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                  continue
               else:
                  # Otherwise move the token_start_index and token_end_index to the two ends of the token.
                  # Note: we could go after the last offset if the answer is the last word (edge case).
                  while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                      token_start_index += 1
                  token_start_index = token_start_index-1
               
                  while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                       token_end_index -= 1
                  if offsets[token_end_index +1 ][1]== end_char:
                       token_end_index = token_end_index+ 2
                  else:
                       token_end_index = token_end_index+ 1
                  # if start and end are on the same token
                  if token_start_index== token_end_index:
                     token_end_index+=1

                  # retrieve list of concept embedding for this token 
                  list_con_ids = [0]* max_concepts_len
                  cons= e['q_concepts']
              
                  not_f=0
                  cnt=0
                  for c in cons:                 
                     try:
                        if cnt < max_concepts_len:
                            list_con_ids[cnt] = self.entity2idx[c]
                            cnt+=1
                            continue
                        else:
                          break
                     except KeyError:
                       not_f+=1
              
                  
              
                  #print(tokenizer.decode(tokenized_examples["input_ids"][i][token_start_index: token_end_index]))
                  #assign concepts embedding to this token 
                  while token_start_index< token_end_index:
                    concepts_ids[token_start_index]=list_con_ids
                    lengths[token_start_index] = len(list_con_ids)
                    token_start_index+=1
          
            # if we have multiple spans        
            c_ids.append(concepts_ids)
            # number of retrieved concepts for each token
            lens.append(lengths)
        
       #prepeare mask for padded concepts 
        features_mask=[]
        for ex in lens:
          sequence=[]
          for tok_len in ex:
             mask=[0]* max_concepts_len
             mask[:tok_len]= [1]* tok_len
             sequence.append(mask)
          features_mask.append(sequence)

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        k=0
        for i in range(len(tokenized_examples["input_ids"])):
          # Grab the sequence corresponding to that example (to know what is the context and what is the question).
          sequence_ids = tokenized_examples.sequence_ids(i)
          context_index = 1 if pad_on_right else 0
       
          # One example can give several spans, this is the index of the example containing this span of text.
          sample_index = sample_mapping[i]
          tokenized_examples["example_id"].append(examples["id"])

          # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
          # position is part of the context or not.
          offset_mapping[i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(offset_mapping[i])]

        input_ids= [ f for f in tokenized_examples['input_ids']]
        attention_masks = [f for f in tokenized_examples['attention_mask']]
        token_type_ids = [f for f in tokenized_examples['token_type_ids']]
        start_positions = [f for f in tokenized_examples['start_positions']]
        end_positions = [f for f in tokenized_examples['end_positions']]
        example_id = [f for f in tokenized_examples["example_id"]]
        offset_mapping = [f for f in offset_mapping ]
        
        features=[]
        # if it is one span it will return one example if multiple spans it wil return multiple examples or features
        for i in range(len(input_ids)):

            feature = {
                'input_ids':[],
                'attention_masks':[],
                'token_type_ids':[],
                'start_positions':int,
                'end_positions':int, 
                'concept_ids':[], 
                'padded_concepts_mask':[]
            }

            feature['input_ids'] = input_ids[i]
            feature['attention_masks'] = attention_masks[i]
            feature['token_type_ids'] = token_type_ids[i]
            feature['start_positions'] = start_positions[i]
            feature['end_positions'] = end_positions[i]
            feature['concept_ids'] = c_ids[i]
            feature['padded_concepts_mask'] = features_mask[i]
            
            features.append(feature)

        return features, offset_mapping, example_id 


  def prepare_validation_features(self, examples, concept):
    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
        
    pad_on_right = self.tokenizer.padding_side == "right"
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = self.tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    #tokenized_examples['offset_mapping'] = tokenized_examples.pop("offset_mapping")

    
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
        
    # embedding for all features
    c_ids = []
    lens=[]
    for i, offsets in enumerate(offset_mapping):
       
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["text"]) == 0 or len(str(answers['answer_start']))==0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while token_start_index < len(sequence_ids)-1 and sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
            

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while token_end_index != 0  and sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
            
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index-1)
               
                    while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                       token_end_index -= 1
                    '''
                    if offsets[token_end_index +1 ][1]== end_char:
                        tokenized_examples["end_positions"].append(token_end_index+ 2)
                    else:
                    '''
                    tokenized_examples["end_positions"].append(token_end_index+ 1)
        
    

    # embedding for all features
    # match concepts
    c_ids = []
    lens=[]
    for i, offsets in enumerate(offset_mapping):
       
       sequence_ids = tokenized_examples.sequence_ids(i)
       max_concepts_len= 40
       #create conceptsE sequence
       concepts_ids= [[0]* max_concepts_len ]*len(sequence_ids)
       lengths= [0]* len(sequence_ids)
            
       # map documents concepts
       ''' 
       subs=[]
       from_index = 0

       if len(concept['doc_entities'])> max_concepts_len:
         concept['doc_entities']= concept['doc_entities'][:max_concepts_len]

       for e in concept['doc_entities']:
         
          if e['entity'].find('_')!= -1 and len(subs)==0:
            subs= e['entity'].split('_')

          # one token 
          if  e['entity'].find('_')==-1:
            try:
                start_char = examples['context'].index(e['entity'], from_index)
                
            except ValueError: 
                    print("couldn't find concept ", e['entity'])
                    continue
            end_char= start_char + len(e['entity'])
            from_index= end_char
          else:

             # correct start and end position 
             try: 
                start_char = examples['context'].index(subs[0], from_index)
             except ValueError: 
                print("couldn't find concept ", subs[0] )
                continue
             end_char= start_char + len(subs[0])
             from_index= end_char
             if len(subs)!=0:
                #delete item in index 0 
                subs.pop(0)

         
          #print(examples['context'][start_char:end_char])
          # Start token index of the current span in the text.
          token_start_index = 0
          while token_start_index < len(sequence_ids)-1 and sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                   token_start_index += 1
             

          # End token index of the current span in the text.
          token_end_index = len(tokenized_examples["input_ids"]) - 1
          while token_end_index != 0  and sequence_ids[token_end_index] != (1 if pad_on_right else 0):
              token_end_index -= 1
             

          # Detect if the concept token is out of the span (do nothing).
          if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                  
               continue
          else:
              # Otherwise move the token_start_index and token_end_index to the two ends of the token.
              # Note: we could go after the last offset if the answer is the last word (edge case).
              while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                  token_start_index += 1
              token_start_index = token_start_index-1
               
              while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                   token_end_index -= 1
                  
              if offsets[token_end_index +1 ][1]== end_char:
                  token_end_index = token_end_index+ 2
              else:
                  token_end_index = token_end_index+ 1
              # if start and end are on the same token
              if token_start_index== token_end_index:
                 token_end_index+=1

              # retrieve list of concept id for this token 
              list_con_ids = [0]* max_concepts_len
              cons= e['concepts']
              
              not_f=0
              cnt=0
              for c in cons:                 
                try:
                  if cnt < max_concepts_len:
                    list_con_ids[cnt] = self.entity2idx[c]
                    cnt+=1
                    continue
                  else:
                     break
                except KeyError:
                    not_f+=1
                  

               #assign concepts ids to this token 
              while token_start_index< token_end_index:
                  concepts_ids[token_start_index]=list_con_ids
                  lengths[token_start_index] = len(list_con_ids)
                  token_start_index+=1

             
        '''      
               # map q concepts
               # TODO: we must check sequence_id
       subs=[]
       from_index = 0
      
       if len(concept['question_entities'])> max_concepts_len:
         concept['question_entities']= concept['question_entities'][:max_concepts_len]

       for  e in concept['question_entities']:
          
          if e['entity'].find('_')!= -1 and len(subs)==0:
              subs= e['entity'].split('_')

          # one token 
          if e['entity'].find('_')==-1:
            try:
               start_char = examples['question'].index(e['entity'], from_index)
            except ValueError: 
               print("couldn't find concept ", e['entity'])
               continue
            end_char= start_char + len(e['entity'])
            from_index= end_char
          else:

            # correct start and end position 
            try:
               start_char = examples['question'].index(subs[0], from_index)
            except ValueError: 
               print("couldn't find concept ", subs[0] )
               continue
            end_char= start_char + len(subs[0])
            from_index= end_char
            if len(subs)!=0:
              #delete item in index 0 
              subs.pop(0)

          #print(examples['question'][start_char:end_char])
          # Start token index of the current span in the text.
          token_start_index = 0
          # End token index of the current span in the text.
          token_end_index = 1
          while token_end_index < len(sequence_ids)-1 and sequence_ids[token_end_index] ==0 :
            token_end_index+=1

          token_end_index-=1  
             

          # Detect if the concept token is out of the span (do nothing).
          if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
             continue
          else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the token.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index = token_start_index-1
               
            while token_end_index != 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            if offsets[token_end_index +1 ][1]== end_char:
                token_end_index = token_end_index+ 2
            else:
                token_end_index = token_end_index+ 1
            # if start and end are on the same token
            if token_start_index== token_end_index:
              token_end_index+=1

            # retrieve list of concept embedding for this token 
            list_con_ids = [0]* max_concepts_len
            cons= e['q_concepts']
              
            not_f=0
            cnt=0
            for c in cons:                 
              try:
                if cnt < max_concepts_len:
                    list_con_ids[cnt] = self.entity2idx[c]
                    cnt+=1
                    continue
                else:
                    break
              except KeyError:
                    not_f+=1
              
                  
              
            #print(tokenizer.decode(tokenized_examples["input_ids"][i][token_start_index: token_end_index]))
            #assign concepts embedding to this token 
            while token_start_index< token_end_index:
                concepts_ids[token_start_index]=list_con_ids
                lengths[token_start_index] = len(list_con_ids)
                token_start_index+=1
          
       # if we have multiple spans        
       c_ids.append(concepts_ids)
       # number of retrieved concepts for each token
       lens.append(lengths)
        
    #prepeare mask for padded concepts 
    features_mask=[]
    for ex in lens:
      sequence=[]
      for tok_len in ex:
          mask=[0]* max_concepts_len
          mask[:tok_len]= [1]* tok_len
          sequence.append(mask)
      features_mask.append(sequence)

       
    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    k=0
    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
       
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        offset_mapping[i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(offset_mapping[i])
        ]
    

    input_ids= [ f for f in tokenized_examples['input_ids']]
    attention_masks = [f for f in tokenized_examples['attention_mask']]
    token_type_ids = [f for f in tokenized_examples['token_type_ids']]
    example_id = [f for f in tokenized_examples["example_id"]]
    offset_mapping = [f for f in offset_mapping ]
    start_positions = [f for f in tokenized_examples['start_positions']]
    end_positions = [f for f in tokenized_examples['end_positions']]
        
    features=[]
    # if it is one span it will return one example if multiple spans it wil return multiple examples or features
    for i in range(len(input_ids)):

            feature = {
                'input_ids':[],
                'attention_masks':[],
                'token_type_ids':[],
                'concept_ids':[], 
                'padded_concepts_mask':[],
                'start_positions':int,
                'end_positions':int 

            }

            feature['input_ids'] = input_ids[i]
            feature['attention_masks'] = attention_masks[i]
            feature['token_type_ids'] = token_type_ids[i]
            feature['concept_ids'] = c_ids[i]
            feature['padded_concepts_mask'] = features_mask[i]
            feature['start_positions'] = start_positions[i]
            feature['end_positions'] = end_positions[i]

            features.append(feature)
    
    return features, offset_mapping, example_id




  def postprocess_qa_predictions(self, examples, features,example_id, offset_mapping, all_start_logits, all_end_logits , n_best_size = 20, max_answer_length = 30):
    
    squad_v2= False
    # Build a map example to its corresponding features.
    example_id_to_index={}
    ids=[]
    for ex in examples:
        ids.append(ex['id'])

    example_id_to_index = {k: i for i, k in enumerate(ids)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[example_id[i]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping_ex = offset_mapping[feature_index]
            len_offset_mapping = len(offset_mapping_ex)
            
            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len_offset_mapping
                        or end_index >= len_offset_mapping
                        or offset_mapping_ex[start_index] is None
                        or offset_mapping_ex[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping_ex[start_index][0]
                    end_char = offset_mapping_ex[end_index][1]
                    
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


 
  def postprocess_qa_predictions_analysis(self, example, features,example_id, offset_mapping, all_start_logits, all_end_logits , n_best_size = 20, max_answer_length = 30):
    
    squad_v2= False
    # Build a map example to its corresponding features.
    example_id_to_index={}
    ids=[]
    examples=[]
    examples.append(example)
    for ex in examples:
        ids.append(ex['id'])

    example_id_to_index = {k: i for i, k in enumerate(ids)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[example_id[i]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping_ex = offset_mapping[feature_index]
            len_offset_mapping = len(offset_mapping_ex)
            
            # Update minimum null prediction.
            input_ids= features[feature_index]["input_ids"]
            # retrieving list of tokens will not work correctly with multiple features
            tokens =  self.tokenizer.convert_ids_to_tokens(input_ids)
            
            cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            # calculate scores
            zipped_lists = zip(start_logits, end_logits)
            scores = [x + y for (x, y) in zipped_lists]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len_offset_mapping
                        or end_index >= len_offset_mapping
                        or offset_mapping_ex[start_index] is None
                        or offset_mapping_ex[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping_ex[start_index][0]
                    end_char = offset_mapping_ex[end_index][1]
                    
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char], 
                            "start_logits": start_logits[start_index] , 
                            "end_logits": end_logits[end_index],
                            "start_index":  start_index, 
                            "end_index": end_index
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            best_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer
        print(tokens)
        print(best_answers[0])
        print(best_answers[1])
        print(best_answers[2])
        print(best_answers[3])
        print(best_answers[4])
        '''
        text= [x['text'] for x in best_answers[:10]]
        start_logits= [x['start_logits'] for x in best_answers[:10]]
        end_logits= [x['end_logits'] for x in best_answers[:10]]
        scores= [x['score'] for x in best_answers[:10]]
        '''
        ar_text=[]
        for t in tokens:
           reshaped_text = arabic_reshaper.reshape(t)
           artext = get_display(reshaped_text)
           ar_text.append(artext)
        
        print(len(ar_text))
        print(len(start_logits))
        print(len(end_logits))
        # plot line chart for start, end logits and score
        #plt.subplots(figsize=(5, 2.7), layout='constrained')
        # plt.plot(ar_text[75:100], start_logits[75:100], label="start_logit", linestyle="--" )
        #plt.plot(ar_text[22:60], scores, label="start_logit", linestyle="-." )
        #plt.plot(ar_text[75:100], end_logits[75:100], label="end_logit", linestyle=":" )
        plt.plot(ar_text[75:100], start_logits[75:100], 'ro')
        plt.plot(ar_text[75:100], end_logits[75:100], 'bs')
        plt.xticks(range(len(ar_text[75:100])), ar_text[75:100], rotation='vertical')
        plt.xlabel('sequence')
        plt.ylabel('score')
        plt.show()

    return predictions
        