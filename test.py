from modeling_cocom import COCOM, COCOMConfig
from utils import prepare_auto_encoding, prepare_text_continuation
from metrics import compute_rouge_scores, exact_match_score 
import numpy as np
from rouge import Rouge
import datasets
import os 
import argparse
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Test Configuration")
    # Model Configuration
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--doc_max_length", type=int, default=128, help="Maximum document length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_test_examples", type=int, default=512, help="number of examples to test")
    parser.add_argument("--dataset_name_or_dir", type=str, default='/local/calmar/rag/datasets/kilt_wikipedia-128_tokens_splits')

    #parser.add_argument("--compression_linear_type", type=str, default='concat')
    args = parser.parse_args()
    return args
    
def collate_batch(batch):
    return_dict = {}
    for key in batch[0]:
        if 'text' not in key:
            return_dict[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            return_dict[key] = [item[key] for item in batch]
    return return_dict


def write(metrics, path):
    with open(path, 'w') as fout:
        fout.write(str(metrics))


def main():
    args = get_args()
    rouge = Rouge()    


    ae_path = f'{args.checkpoint.split("/last_model")[0]}/{args.dataset_name_or_dir.split("/")[-1]}_ae_eval.txt'
    tc_path = f'{args.checkpoint.split("/last_model")[0]}/{args.dataset_name_or_dir.split("/")[-1]}_tc_eval.txt'
    print(ae_path)
    print(tc_path)
    
    if os.path.exists(ae_path) and os.path.exists(tc_path):
        print(f'Files {ae_path} and {tc_path} exists. Skipping')
        return
    model = COCOM.from_pretrained(args.checkpoint)


    if os.path.exists(args.dataset_name_or_dir):
        dataset = datasets.load_from_disk(args.dataset_name_or_dir)
    else:
        dataset = datasets.load_dataset(args.dataset_name_or_dir)
    dataset = dataset['test'].select(range(args.num_test_examples))

    for pretrain_tokenize_function in [prepare_auto_encoding, prepare_text_continuation]:
        if pretrain_tokenize_function ==  prepare_auto_encoding:
            out_file = ae_path
        else:
            out_file = tc_path
        if os.path.exists(out_file):
            print(f'File {out_file} exists. Skipping')
            continue
        print(f'Testing on {args.num_test_examples} examples.')
        dataset_task = dataset.map(pretrain_tokenize_function, batched=True, load_from_cache_file=False,
                                            fn_kwargs={"compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
                                                        "decoder_tokenizer": model.decoder_tokenizer,
                                                        "enc_max_len": args.doc_max_length,
                                                        "compression_rate": model.compr_rate,
                                                        'train': False,
                                                        }
                                                        )
        
        dataloader = torch.utils.data.DataLoader(dataset_task, args.batch_size, collate_fn=collate_batch)
        model.eval()
        model = model.to('cuda')
        preds, texts, next_texts = list(), list(), list()
        for batch in tqdm(dataloader):
            text = batch.pop('text')
            next_text = batch.pop('next_text')
            pred = model.generate(batch, max_new_tokens=256)
            preds += pred
            next_texts += next_text
            texts += text
        #texts, preds, next_texts = sum(texts, []), sum(preds, []), sum(next_texts, [])
        random_idx = np.random.randint(0, len(texts))
        print(f'Example: {texts[random_idx]}')
        print(f'Prediction: {preds[random_idx]}')
        print(f'Next Text: {next_texts[random_idx]}')
        if pretrain_tokenize_function ==  prepare_auto_encoding:
            rouge_scores = compute_rouge_scores(rouge, preds, texts)
            em = exact_match_score(preds, texts)
            print('Auto Encoding:')
            metrics = {'EM': em}
            metrics.update(rouge_scores)
            write(metrics, out_file)
        else:
            rouge_scores = compute_rouge_scores(rouge, preds, next_texts)
            em = exact_match_score(preds, next_texts)
            print('Text Continuation:')
            metrics = {'EM': em}
            metrics.update(rouge_scores)
            write(metrics, out_file)
        print(metrics)
if __name__ == "__main__": 
    main()
