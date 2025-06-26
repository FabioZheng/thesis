from transformers import AutoTokenizer
import datasets

folder = "/scratch-shared/drau/"

dataset = datasets.load_dataset('kilt_wikipedia')['full']
doc_max_length = 128
num_proc = 20

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=True)

def chunk_text(entry, tokenizer, max_length):
    docs = list()
    for x in entry["text"]:
        passages = [p.strip() for p in x['paragraph'] if "BULLET::::" not in p  ]
        doc = " ".join(passages)
        doc = doc.replace('Section::::', 'Section:')
        docs.append(doc)
    batch_tokens = tokenizer(docs)['input_ids']
    chunks = list()
    for tokens in batch_tokens :
        chunk = tokenizer.batch_decode([tokens[i:i+max_length] for i in range(0, len(tokens), max_length) ][:-1], skip_special_tokens=True)
        chunks.append(chunk)
    return {'chunks': chunks}   

new_dataset = dataset.map(lambda x: chunk_text(x, tokenizer, doc_max_length), remove_columns=['text'], num_proc=num_proc, batched=True)

chunks = [el for sublist in new_dataset['chunks'] for el in sublist[:-1] if len(sublist) > 1]
chunks_next = [el for sublist in new_dataset['chunks'] for el in sublist[1:] if len(sublist) > 1]

# print(len(chunks), len(chunks_next))
# print("chunk 0", chunks[0])
# print("chunk next 0", chunks_next[0])
#
# print("-----------------")
# print("chunk -1", chunks[-1])
# print("chunk next -1", chunks_next[-1])
dataset = datasets.Dataset.from_dict({'text': chunks, "next_text": chunks_next})
#dataset.save_to_disk(folder + f'fineweb-sample-10BT-{doc_max_length}_tokens/')
dataset = dataset.train_test_split(test_size=0.00005)
dataset.save_to_disk(folder + f'/kilt_wikipedia-{doc_max_length}_tokens_splits/')