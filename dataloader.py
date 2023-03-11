#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:41:59 2021

@author: emanalbilali
"""

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
import numpy as np
import torch.nn.functional as F


class DatasetArabicTyDiQA(Dataset):
    def __init__(self, data  ):
      self.data =data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

       data_point= self.data[index]
       all_input_ids= data_point[0]
       all_attention_masks = data_point [1]
       all_token_type_ids = data_point[2]
       all_start_positions = data_point[3]
       all_end_positions= data_point[4]
       all_concept_ids = data_point[5]
       all_padded_concepts_mask = data_point[6]
       

       return all_input_ids, all_attention_masks,all_token_type_ids, all_start_positions,all_end_positions, all_concept_ids, all_padded_concepts_mask
       

class DataLoaderTyDiQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderTyDiQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn