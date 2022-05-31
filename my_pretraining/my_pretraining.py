# coding=utf-8
## create by anthonyluosen base on nvidia bert source code:
## link:aluosen@gmail.com

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
from operator import mod
import os
import time
import argparse
import random
import logging
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
from utils import prepare_training_data
import modeling
from schedulers import PolyWarmUpScheduler
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process, format_step, get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler
# import dllogger
import signal
from torch.utils.tensorboard import SummaryWriter



def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)

# dllogger.init(backends=[])
class BertPretrainingCriterion(torch.nn.Module):

    # sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda", 0)
        args.n_gpu = 1 # torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        if args.cuda_graphs :
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = 1

    print("device: {} n_gpu: {}, distributed training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device, sequence_output_is_dense):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForPreTraining(config, sequence_output_is_dense=sequence_output_is_dense)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location=device)
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)

    # If allreduce_post_accumulation_fp16 is not set, Native AMP Autocast is
    # used along with FP32 gradient accumulation and all-reduce
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.Adam(params=optimizer_grouped_parameters,lr=args.learning_rate)
    # optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
    #                          lr=args.learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=args.init_loss_scale, enabled=args.fp16)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       base_lr=args.learning_rate,
                                       device=device)

    model.checkpoint_activations(args.checkpoint_activations)

    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)

    if args.resume_from_checkpoint and args.init_checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    if args.resume_from_checkpoint:
        # For phase2, need to reset the learning rate and step count in the checkpoint
        if 'grad_scaler' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        # optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)
    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion, start_epoch,grad_scaler

def checkpoint_step(args, epoch, global_step, model,grad_scaler,optimizer,id):
    torch.cuda.synchronize()
    if is_main_process() and not args.skip_checkpoint:
        # Save a trained model
        # dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        
        output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
      
        if args.do_train:
            torch.save({'model': model_to_save.state_dict(),
                        'grad_scaler': grad_scaler.state_dict(),
                        'epoch': epoch,
                        'id':id,
                        'optimizer':optimizer.state_dict()}, output_save_file,
                       )
        


def take_training_step(args, model, criterion, batch,device,grad_scaler,test = False):
    # prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], masked_lm_labels=batch['labels'])
    # loss = criterion(prediction_scores, seq_relationship_score, batch['labels'], batch['next_sentence_labels'])
    with torch.cuda.amp.autocast(enabled=(args.fp16)) :
        all_input_ids  = batch[0].to(device)
        all_input_mask = batch[1].to(device) 
        all_segment_ids= batch[2].to(device)
        all_label_ids  = batch[3].to(device)
        next_sentence_labels = batch[4].to(device)
        prediction_scores, seq_relationship_score = model(input_ids=all_input_ids, token_type_ids=all_segment_ids,
                attention_mask=all_input_mask, masked_lm_labels=all_label_ids)
        loss = criterion(prediction_scores, seq_relationship_score, all_label_ids,
                next_sentence_labels) 
    # grad_scaler.scale(loss).backward()
        # loss.backward()
        if test:
            return loss
        grad_scaler.scale(loss).backward()
    return loss


def take_optimizer_step(lr_scheduler, optimizer,grad_scaler):
    lr_scheduler.step()  # learning rate warmup
    grad_scaler.step(optimizer)
    optimizer.step()
    # Stats copying is located here prior to the infinity check being reset
    # in GradScaler::update()
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .h5py files for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=True,
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
   
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
 
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of DataLoader worker processes per rank')
    parser.add_argument('--tb_dir',
                        type=str,
                        default='./runs',
                        help='log the training info')
    
    # optimizations controlled by command line arguments

    args = parser.parse_args()


    return args

def main():
    global timeout_sent
    
    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    device, args = setup_training(args)
    # dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, criterion, start_epoch,grad_scaler = prepare_model_and_optimizer(args, device, sequence_output_is_dense=False)
    # Prepare the data loader.
    # if is_main_process():
    #     tic = time.perf_counter()

    # if is_main_process():
    #     print('get_bert_pretrain_data_loader took {} s!'.format(time.perf_counter() - tic))
    # trainloader = prepare_training_data(2,num_jobs=args.num_workers,path=args.input_dir,type_='training',
    # device='cpu',train_batch_size=args.train_batch_size,dense_output=False,)
    # print(len(trainloader))
    # print(len(trainloader)/args.train_batch_size)
    

    # if is_main_process():
    #     dllogger.log(step="PARAMETER", data={"SEED": args.seed})
    #     dllogger.log(step="PARAMETER", data={"train_start": True})
    #     dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
    #     dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})
    tbw = SummaryWriter(log_dir=args.tb_dir) # Tensorboard logging
    model.train()
    seen = 0
    raw_train_start = None
    epoch = 0
    tr_loss = []
    id = 0
    while True:
        trainloader = prepare_training_data(id,num_jobs=args.num_workers,path=args.input_dir,type_='training',
    device='cpu',train_batch_size=args.train_batch_size,no_dense_output=True,)
        train_iter = tqdm(
        trainloader,
        desc="Iteration",
        disable=args.disable_progress_bar,
        total=len(trainloader),
    ) if is_main_process() else trainloader
        testloader = prepare_training_data(id,num_jobs=args.num_workers,path=args.input_dir,type_='test',
    device='cpu',train_batch_size=6,no_dense_output=True,)
        id += 1
        for step, batch in enumerate(train_iter):
            global_step += 1
            loss = take_training_step(args, model, criterion, batch, device,grad_scaler=grad_scaler)
            tr_loss.append(loss.item())
            take_optimizer_step(lr_scheduler, optimizer,grad_scaler)
            if step % 1000 == 0:
                print(np.array(tr_loss).mean())
                tr_loss = []
            epoch +=1
            tbw.add_scalar('mlm_next/train-loss', float(loss.item()), global_step)
        with torch.no_grad():
            test_loss = []
            for step_, batch_ in enumerate(testloader):
                loss_ = take_training_step(args, model, criterion, batch_, device,grad_scaler=grad_scaler,
                                           test=True)
                test_loss.append(loss_.item())
        tbw.add_scalar('mlm_next/test-loss', float(np.array(test_loss).mean()), global_step)
        checkpoint_step(args, epoch, global_step=global_step, model=model,grad_scaler=grad_scaler,
                        optimizer=optimizer,id=id)

if __name__ == "__main__":

    now = time.time()
    main()
    
   



