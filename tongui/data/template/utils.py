import pdb
import torch
# from .data_utils import replace_image_tokens

"""
(multi-turn) template: 

<|user|>
system_prompt<|image_1|>query<|end|>
<|assistant|>
answer<|endoftext|>
<|user|>
query2<|end|>
<|assistant|>
answer2<|endoftext|>
<|user|>
query3<|end|>
<|assistant|>
answer3<|endoftext|>

example:

<|user|>
I'll give textual clues about elements in this webpage screenshot; identify their coordinates.<|image_1|>Feb182019<|end|>
<|assistant|>
 [0.02, 0.86]<|endoftext|>
<|user|>
Conveyancing Team<|end|>
<|assistant|>
[0.08, 0.37]<|endoftext|>
<|user|>
Admin Team<|end|>
<|assistant|>
[0.07, 0.64]<|endoftext|>
<|user|>
Facebook page opens in new window<|end|>
<|assistant|>
"""

def batch_add_answer(batch, answer, processor, add_eos=True):
    prompt_input_ids = batch['input_ids']

    if not isinstance(answer, str):
        answer = str(answer)
    if add_eos:
        answer += processor.tokenizer.eos_token
    answer_input_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors='pt'
    )['input_ids']
    input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
    ignore_index = -100
    labels = torch.cat(
        [
            torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
            answer_input_ids,
        ],
        dim=1,
    )

    batch['input_ids'] = input_ids
    del batch['attention_mask']
    batch['labels'] = labels

    return batch, answer

def batch_add_answer_append(batch, answer, processor, add_eos=True, 
                            append_element=None, append_answer=None,
                            add_generation_prompt=True):
    prompt_input_ids = batch['input_ids']

    if not isinstance(answer, str):
        answer = str(answer)
    if add_eos:
        answer += processor.tokenizer.eos_token
    answer_input_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors='pt'
    )['input_ids']
    input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
    ignore_index = -100
    labels = torch.cat(
        [
            torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
            answer_input_ids,
        ],
        dim=1,
    )

    # multi-turn.
    append = answer
    for i, (element_i, answer_i) in enumerate(zip(append_element, append_answer)):
        if not isinstance(answer_i, str):
            answer_i = str(answer_i)
        if not isinstance(element_i, str):
            element_i = str(element_i)
        if add_eos:
            answer_i += processor.tokenizer.eos_token

        if add_generation_prompt:
            source_i = [{"role": "user", "content": element_i}]
            element_i = '\n' + processor.tokenizer.apply_chat_template(source_i, chat_template=processor.chat_template, tokenize=False, add_generation_prompt=True)

        append += element_i + answer_i
        element_input_ids = processor.tokenizer(
            element_i, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        answer_input_ids = processor.tokenizer(
            answer_i, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([input_ids, element_input_ids, answer_input_ids], dim=1)
        labels = torch.cat(
                        [
                            labels, 
                            torch.tensor([ignore_index] * len(element_input_ids[0])).unsqueeze(0),
                            answer_input_ids,
                        ],
                        dim=1,
        )

    batch['input_ids'] = input_ids
    del batch['attention_mask']
    batch['labels'] = labels
    return batch, append