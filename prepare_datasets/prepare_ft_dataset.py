import datasets

train_data = list()



# triviaqa 
# nq_open
# wikiqa
# yahoo_qa
# commonsense_qa
# webqa
# freebase_qa
# ms_marco
# adverserial_qa


# nq_open
dataset = datasets.load_dataset("nq_open")
for idx,sample in enumerate(dataset['train']):
    messages = []
    answer = sample['answer'][0]
    question = sample['question']
    context = None
    id_ = f'nq_open{idx}'
    train_data.append((id_, question, answer))
print('nq_open', train_data[-1])

# ms_marco
data = datasets.load_dataset('ms_marco',"v2.1")['train']
for idx,sample in enumerate(data.select(range(100000))):
    question = sample['query']
    id_ = f'msmarco{idx}'
    answer = sample["answers"][0]
    train_data.append((id_, question, answer))
print('msmarco', train_data[-1])




# adverserial_qa
dataset = datasets.load_dataset("UCLNLP/adversarial_qa", "adversarialQA")
for idx,sample in enumerate(dataset['train']):
    messages = []
    answer = sample['answers']['text'][0]
    question = sample['question']
    context = sample['context']
    id_ = f'adversarial_qa{idx}'
    train_data.append((id_, question, answer))
print('adverserial qa', train_data[-1])

#hotpotqa
dataset = datasets.load_dataset("kilt_tasks", "hotpotqa")
for idx,sample in enumerate(dataset['train']):
    messages = []
    answer = sample['output'][0]['answer']
    question = sample['input']
    context = None
    id_ = f'hotpotqa{idx}'
    train_data.append((id_, question, answer))

print('hotpot_qa', train_data[-1])


#wikiqa
dataset = datasets.load_dataset('wiki_qa')['train']
# discarding empty answers 
dataset_f = dataset.filter (lambda x: x['label'] == 1) # keeping only the valid sentences

# ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
# No ranking labels
#dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})

dataset_l=[]
qid_set=set(dataset_f['question_id'])

for idx, q in enumerate(qid_set):
    qsel= dataset_f.filter(lambda x: x['question_id']==q)
    id_ = f'wikiqa{idx}'
    question = qsel['question'][0]
    answer = qsel['answer']
    train_data.append((id_, question, answer))
print('wiki qa', train_data[-1])


#sciq
dataset = datasets.load_dataset('sciq')['train']
# query
dataset = dataset.rename_column("question", "content")
dataset = dataset.map(lambda example: {'label': [example['correct_answer']]})
dataset = dataset.remove_columns(["support","correct_answer","distractor1", "distractor2","distractor3"])        
for idx, item in enumerate(dataset):
    id_ = f'sciq{idx}'
    question = item['content']
    answer = item['label']
    train_data.append((id_, question, answer))
print('sciq', train_data[-1])

# asqa
hf_name = 'din0s/asqa' 
dataset = datasets.load_dataset(hf_name)['train']

dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)
dataset = dataset.rename_column("ambiguous_question", "content")

# get short answers
def short_answers(example):
    z = list(set([ ans for qa in example['qa_pairs']  for ans in qa['short_answers'] ]))
    return z 

dataset = dataset.map(lambda example: {'label': short_answers(example)})
#dataset = dataset.map(lambda example: {'ranking_label': get_wiki_id(example)},num_proc=5)

dataset = dataset.remove_columns([ 'qa_pairs', 'wikipages', 'annotations', 'sample_id'])
for idx, item in enumerate(dataset):
    id_ = f'asqa{idx}'
    question = item['content']
    answer = item['label']
    train_data.append((id_, question, answer))
print('asqa', train_data[-1])


# triviaqa
dataset = datasets.load_dataset("kilt_tasks", name="triviaqa_support_only")['train']
hf_q_ids = set(dataset['id'])
trivia_qa = datasets.load_dataset('trivia_qa', 'unfiltered.nocontext')['train']
def add_missing_data(x, trivia_qa_subset, triviaqa_map):
    i = triviaqa_map[x['id']]
    x['input'] = trivia_qa_subset[i]['question']
    x['output'][0]['original_answer'] = trivia_qa_subset[i]['answer']['value']
    return x
    
triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa['question_id'])])
dataset = dataset.filter(lambda x: x['id'] in triviaqa_map)
# only use ids that are present in the kilt_dataset
dataset = dataset.filter(lambda x: x['id'] in hf_q_ids)
dataset = dataset.map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa, triviaqa_map=triviaqa_map))

# discarding empty answers 
dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
# ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
dataset = dataset.rename_column("input", "content")
dataset = dataset.remove_columns(['meta', 'output'])
for idx, item in enumerate(dataset):
    id_ = f'triviaqa{idx}'
    question = item['content']
    answer = item['label']
    train_data.append((id_, question, answer))
print('wiki_qa', train_data[-1])


# freebase_qa
data = datasets.load_dataset('freebase_qa')
for idx,sample in enumerate(data['train']):
    question = sample['RawQuestion']
    id_ = f'freebase_qa{idx}'
    answer = sample["Parses"]['Answers'][0]['AnswersName'][0][0]
    train_data.append((id_, question, answer))
print('freebase', train_data[-1])



# # -----------------------------------------------------------------------------
# # yahoo_qa
# data = datasets.load_dataset('yahoo_answers_qa')
# for idx,sample in enumerate(data['train']):
#     question = sample['question']
#     answer = sample['answer']
#     id_ = f'yahoo_answers_qa{idx}'
#     train_data.append((id_, question, answer))
# print('yahoo', train_data[-1])

# squad_v1.1
data = datasets.load_dataset("squad")
for idx,sample in enumerate(data['train']):
    if len(sample['answers']['text']) == 0:
        continue
    question = sample['question']
    answer = sample['answers']['text']
    context = sample['context']
    id_ = f'squad{idx}'
    train_data.append((id_, question, answer))

print('squad', train_data[-1])



column_names = ['id', 'content', 'label']
data_dict = {col: [x[i] for x in train_data] for i, col in enumerate(column_names)}

data_dict['label'] = [[label] if isinstance(label, str) else label for label in data_dict['label']]
ft_dataset = datasets.Dataset.from_dict(data_dict)
ft_dataset = ft_dataset.shuffle(seed=42)
ft_dataset.save_to_disk('qa_dataset.hf')
print(ft_dataset)




# templates_for_sum = [
#     "Write a short summary for the text\n\nSummary:",
#     "Briefly summarize this article:\nSummary:", 
#     "What is a shorter version of this:\n\nSummary:",
#     "Write a brief summary in a sentence or less.", 
#     "What is a very short summary of the above text?",
#     "Summarize the aforementioned text in a single phrase.",
#     "Can you generate a short summary of the above paragraph?",
#     "Summarize the above articles\n\ntl;dr:",
# ]

# multiple choice

# commonsense_qa
# data = datasets.load_dataset("commonsense_qa")['train']
# for idx, sample in enumerate(data):
#     question = sample['question']
#     for choice,text in zip(sample['choices']['label'],sample['choices']['text']):
#         question += choice + ". " + text + '\n'
#     id_ = f'commonsense_qa_{idx}'
#     print(sample)
#     answer = sample['answerKey']
#     train_data.append((id_, question, answer))


# # webqa
# data = datasets.load_dataset('web_questions')['train']
# for idx,sample in enumerate(data):
#     question = sample['question']
#     answer = sample['answers'][0]
#     id_ = f'web_questions{idx}'
#     print(sample)
#     answer = sample['answerKey']

#     train_data.append((id_, question, answer))

# # quail multi choice
# data = datasets.load_dataset("quail")
# for idx,sample in enumerate(data['train']):
#     question = sample['question'] + '\n'
#     for answer_id,answer in enumerate(sample['answers']):
#         question += ["A. ","B. ","C. ","D. "][answer_id]+answer+'\n'
#     answer = "The correct answer is" + ["A","B","C","D"][sample["correct_answer_id"]] + '.'
#     context = sample['context']
#     id_ = f'quail{idx}'
#     train_data.append((id_, question, answer))


# with context
 # too long let's exlude for now
# pubmed
# data = datasets.load_dataset("pubmed_qa","pqa_labeled")
# for idx,sample in enumerate(data['train']):
#     question = sample['question']
#     answer = sample['long_answer']
#     context = sample['context']['contexts']
#     id_ = f'pubmed_qa{idx}'
#     train_data.append((id_, question, answer))




# summarization

# # dailymail
# data = datasets.load_dataset("cnn_dailymail",'3.0.0')['train']
# for idx,sample in enumerate(data[:10000]):
#     print(sample)
#     answer = sample['highlights']
#     question = random.choice(templates_for_sum)
#     id_ = f'cnn_dailymail{idx}'
#     context = sample['article']
#     train_data.append((id_, question, answer))

# samsum
# dataset = datasets.load_dataset("samsum")
# for idx,sample in enumerate(dataset['train']):
#     messages = []
#     answer = sample['summary']
#     question = random.choice(templates_for_sum)
#     context = sample['dialogue'].replace("\r\n",'\n')
#     id_ = f'samsum{idx}'
#     if len(context) > 0:

#         train_data.append((id_, question, answer))

# #dialogsum
# dataset = datasets.load_dataset("knkarthick/dialogsum")
# for idx,sample in enumerate(dataset['train']):
#     answer = sample['summary']
#     question = random.choice(templates_for_sum)
#     context = sample['dialogue']
#     id_ = f'dialogsum{idx}'
#     train_data.append((id_, question, answer))

# # drop
# data = datasets.load_dataset("drop")
# for idx,sample in enumerate(data['train']):
#     messages = []
#     question = sample['question']
#     answer = sample["answers_spans"]['spans'][0]
#     context = sample['passage']
#     id_ = f'drop{idx}'
#     train_data.append((id_, question, answer))

# # coqa
# data = datasets.load_dataset("coqa")
# for idx,sample in enumerate(data['train']):
#     assert len(sample['answers']['input_text']) == len(sample['questions'])
#     for idx,(q,a) in enumerate(zip(sample['questions'],sample['answers']['input_text'])):
#         question = q 
#         answer = a
#         context = sample['story']
#         id_ = f'coqa{idx}'
#         train_data.append((id_, question, answer))


# # # eli5
# # dataset = datasets.load_dataset("kilt_tasks", "eli5")
# # for idx,sample in enumerate(dataset['train']):
# #     messages = []
# #     answer = sample['output'][0]['answer']
# #     question = sample['input']
# #     context = None
# #     id_ = f'adversarial_qa{idx}'
# #     train_data.append((id_, question, answer))



