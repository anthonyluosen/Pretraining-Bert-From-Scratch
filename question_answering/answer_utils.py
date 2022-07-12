from data import read_squad_examples,convert_examples_to_features
from tokenization import BertTokenizer
tokenizer = BertTokenizer('.\\vocab\\vocab', do_lower_case=True, max_len=512) # for bert large

import modeling
import torch
config = modeling.BertConfig.from_json_file('bert_pretrain\\bert_config.json')
# Padding for divisibility by 8
# if config.vocab_size % 8 != 0:
#     config.vocab_size += 8 - (config.vocab_size % 8)

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
model = modeling.BertForQuestionAnswering(config)
# checkpoint = torch.load('bert_pretrain\pytorch_model.bin', map_location='cpu')
# checkpoint = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
# model.load_state_dict(checkpoint, strict=False)
model.to(device)

from tqdm import tqdm
import collections
from torch.utils.data import TensorDataset,SequentialSampler,DataLoader

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

eval_examples = read_squad_examples(
    input_file='squad\\v1.1\dev-v1.1.json', is_training=False, version_2_with_negative=False)
eval_features = convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=126,
    max_query_length=60,
    is_training=False)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=4)


# model = model.to(device=device)
model.eval()
all_results = []
for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
    for i, example_index in enumerate(example_indices):
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = eval_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(unique_id=unique_id,
                                    start_logits=start_logits,
                                    end_logits=end_logits))

