# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from pathlib import Path
import os
from joblib import Parallel, delayed, parallel_backend
import h5py
from tqdm import tqdm
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id,next_sentence_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.next_sentence_labels = next_sentence_labels
def _prepare_training_data_helper(type_,train_ids,path,dense_output = False):
    training_samples = []
    for id in tqdm(train_ids):
        path1 = os.path.join(path,type_+'_'+str(id)+'.hdf5')
        f = h5py.File(path1, 'r')
        length = len(f['input_ids'])
    
        masked_lm_positions = np.array(f['masked_lm_positions'])
        masked_lm_ids = np.array(f['masked_lm_ids'])
        input_ids = np.array(f['input_ids'])
        input_mask = np.array(f['input_mask'])
        segment_ids = np.array(f['segment_ids'])
        next_sentence_labels = np.array(f['next_sentence_labels'])
        max_seq_length = input_mask.shape[1]
        if dense_output:
            for data in range(length):
                lm_label_array = np.full(max_seq_length, dtype=int, fill_value=-1)
                lm_label_array[masked_lm_positions[data]] = masked_lm_ids[data]
                feature = InputFeatures(input_ids=input_ids[data],
                                    input_mask=input_mask[data],
                                    segment_ids=segment_ids[data],
                                    label_id=lm_label_array,
                                    next_sentence_labels =next_sentence_labels[data] )
                training_samples.append(feature)
        else:
            for data in range(length):
                feature = InputFeatures(input_ids=input_ids[data],
                                    input_mask=input_mask[data],
                                    segment_ids=segment_ids[data],
                                    label_id=masked_lm_ids[data],
                                    next_sentence_labels =next_sentence_labels[data])
            # feature = {
            #     'input_ids' : f['input_ids'][data],
            #     'input_mask':f['input_mask'][data],
            #     'segment_ids':f['segment_ids'][data],
            #     'label_id':lm_label_array
            # }
                training_samples.append(feature)
    return training_samples

def prepare_training_data(length,num_jobs,path,type_,device,train_batch_size,no_dense_output,parallel=False):
    '''
    length:define the file nums to use as training 
    num_jobs :n jobs to processe the data
    '''
    training_samples = []
    train_ids = np.arange(length)

    train_ids_splits = np.array_split(train_ids, num_jobs)
    if parallel:
        results = Parallel(n_jobs=num_jobs)(
            delayed(_prepare_training_data_helper)(type_, idx, path,no_dense_output) for idx in train_ids_splits
        )
        for result in results:
            training_samples.extend(result)
    else:
        # for idx in train_ids:
        idx = [length]
        results = _prepare_training_data_helper(type_, idx, path,no_dense_output)
        for result in results:
            training_samples.append(result)
    print()
    all_input_ids = torch.tensor(np.array([f.input_ids for f in training_samples]), dtype=torch.int64)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in training_samples]), dtype=torch.int64)
    all_segment_ids = torch.tensor(np.array([f.segment_ids for f in training_samples]), dtype=torch.int64)
    all_label_ids = torch.tensor(np.array([f.label_id for f in training_samples]), dtype=torch.int64)
    next_sentence_labels = torch.tensor(np.array([f.next_sentence_labels for f in training_samples]), dtype=torch.int64)
    # all_input_ids = np.array([f.input_ids for f in training_samples])
    # all_input_mask = np.array([f.input_mask for f in training_samples])
    # all_segment_ids = np.array([f.segment_ids for f in training_samples])
    # all_label_ids = np.array([f.label_id for f in training_samples])
    # next_sentence_labels = np.array([f.next_sentence_labels for f in training_samples])
    
    # all_input_ids = torch.tensor([f.input_ids for f in training_samples], dtype=torch.long, device=device)
    # all_input_mask = torch.tensor([f.input_mask for f in training_samples], dtype=torch.long, device=device)
    # all_segment_ids = torch.tensor([f.segment_ids for f in training_samples], dtype=torch.long, device=device)
    # all_label_ids = torch.tensor([f.label_id for f in training_samples], dtype=torch.long, device=device)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
    next_sentence_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,batch_size=train_batch_size,sampler=train_sampler)
    return train_dataloader



def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def mkdir_by_main_process(path):
    if is_main_process():
        mkdir(path)
    barrier()
