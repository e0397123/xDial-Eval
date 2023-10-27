import torch
import argparse
import json
import logging
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM as LlamaForCausalLM, AutoTokenizer as LlamaTokenizer
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

def get_raw_data(inp_file, lang):
    data = pd.read_csv(inp_file)
    contexts = list(data[f'{lang}_ctx'])
    responses = list(data[f'{lang}_res'])
    prompts = [{'prompt': f"Context:\n{c}\nResponse:\n{r}"} for c, r in zip(contexts, responses)]                                                                                                                                                  
    return prompts, list(data['ratings'])


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data_folder", default='data/dial/', type=str)
    parser.add_argument("--output_folder", default='outputs/', type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--max_encode_len', type=int, default=256)
    return parser.parse_args()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_outputs_withprobs(args, model, tokenizer, batch):
    texts = [d['prompt'] for d in batch]
    new_texts = []
    for t in texts:
        ####Baichuan-2-template
        p = f"<reserved_106>Given the context and response, predict whether the response is relevant to the context?\n\n{t}\n\n<reserved_107>"""
        new_texts.append(p)
    encoding = tokenizer(new_texts, return_tensors='pt', truncation=True, max_length=args.max_encode_len).to(model.device)
    label_list_str = ['Yes', 'No']
    label_list = [tokenizer.encode(x)[0] for x in label_list_str]
    with torch.no_grad():
        generated = model.generate(**encoding, do_sample=False, return_dict_in_generate=True, output_scores =True, max_new_tokens=1)
        generated_scores = generated['scores'][0]
        norm_scores = torch.softmax(generated_scores,dim=1)
        label_probs_list = []
        for label in label_list:
            probs = torch.index_select(norm_scores,1,torch.tensor([label]).cuda())[:,0].tolist()
            label_probs_list.append(probs)
        sum_prob_list = [sum(row[i] for row in label_probs_list) for i in range(len(label_probs_list[0]))]
        balanced_label_probs_list = [[label_probs_list[r][c]/sum_prob_list[c] for c in range(len(label_probs_list[r]))] for r in range(len(label_probs_list))]

    generated_texts = tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)
    transpose_balanced_label_probs_list = [list(i) for i in zip(*balanced_label_probs_list)]

    return generated_texts, transpose_balanced_label_probs_list

def run_generation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    tokenizer.truncation_side = 'left'
    model = LlamaForCausalLM.from_pretrained(args.model, device_map='auto', trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device)
    
    input_f_list = ['conture-turn.csv', 'convai2-grade.csv', 'dailydialog-grade.csv', 'dailydialog-gupta.csv', 'dailydialog-zhao.csv', 
                    'dstc10-persona_clean.csv', 'dstc10-topical_clean.csv', 'empathetic-grade.csv', 'fed-turn.csv', 'persona-usr.csv', 
                    'persona-zhao.csv', 'topical-usr.csv']
    
    for input_file in input_f_list:
        input_file_full_path = os.path.join(args.data_folder, input_file)
        inference_data, infernce_ratings = get_raw_data(input_file_full_path, args.lang)
        
        batch_chunks = chunks(inference_data, args.batch_size)
        input_file_name = input_file_full_path.split('/')[-1].split('.')[0] + '.jsonl'

        os.makedirs(args.output_folder, exist_ok=True)
        model_scores = []       
        with open(os.path.join(args.output_folder, input_file_name), 'w') as outfile:
            for b, batch in tqdm(enumerate(batch_chunks), total=len(inference_data)//args.batch_size):
                try:
                    outputs, probs = get_outputs_withprobs(args, model, tokenizer, batch)
                except:
                    outputs = ['error']
                    probs = [[0.0, 1.0]]
                    model_scores.append(0.0)
                # print(outputs)
                for d, dp in enumerate(batch):
                    batch[d]['output'] = outputs[d]
                    batch[d]['class_probability'] = probs[d]
                    model_scores.append(probs[d][0])
                    # if b%100==0:
                    #     print(batch[d]['output'], batch[d]['class_probability'])
                    json.dump(batch[d], outfile)
                    outfile.write('\n')
                    outfile.flush()
        pearson_score = pearsonr(model_scores, infernce_ratings)
        spearman_score = spearmanr(model_scores, infernce_ratings)
        print(f'Pearson Correlation for {args.lang} - {input_file} is {pearson_score[0]} with p-value {pearson_score[1]}')
        print(f'Spearman Correlation for {args.lang} - {input_file} is {spearman_score[0]} with p-value {spearman_score[1]}')

if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)
    run_generation(args)
