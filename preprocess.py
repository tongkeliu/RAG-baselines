import os
import gzip
import json
from collections import defaultdict
from tqdm import tqdm

data_dir = "/root/autodl-pub/datasets/Amazon-review"
# filename = "Books.jsonl"

# reviewers = defaultdict(list)
# with open(os.path.join(data_dir, filename), "r", encoding='utf-8') as f:
#     for line in f:
#         review = json.loads(line)
#         reviewers[review["user_id"]].append(review)

# for reviewer, reviews in tqdm(reviewers.items()):
#     if len(reviews) >= 200:
#         savename = reviewer
#         with open(os.path.join(data_dir, savename), "w") as f:
#             json.dump(reviews[:200], f, indent=2)
# else:
#     raise FileNotFoundError

# meta_dict = defaultdict(dict)
# meta_filename = 'meta_Books.jsonl.gz'
# with gzip.open(os.path.join(data_dir, meta_filename),"rt") as f:
#     for line in f:
#         data = json.loads(line)
#         meta_dict[data["parent_asin"]] = data

data_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
output_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews_meta"
# with open(os.path.join(data_dir, filename), 'r') as fin:
#     with open(os.path.join(output_dir, filename), "w") as fout:
#         datas = json.load(fin)
#         for data in datas:
#             json.dump(meta_dict[data["parent_asin"]], fout)
#             fout.write('\n')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='cuda', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', trust_remote_code=True)

def generate(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return output

data_collection = []
with open(os.path.join(output_dir, filename), "r") as f:
    for line in f:
        data = json.loads(line)
        if not data['author'] or not data['author']['name']:
            data['author'] = {"name":"Unknown"}
        if not data['title']:
            data['book_summary'] = "None"
            break

        prompt = f'''You are a well-read reader. Given a book title, summarize the book's general content as briefly as possible. 
                book title: {data['title']}, book author: {data['author']['name']}. Please write a summary of the book.'''
        book_summary = generate(prompt)
        data["book_summary"] = book_summary
        data_collection.append(data)
        if len(data_collection) % 10 == 0:
            print("summary finished", len(data_collection))

with open(os.path.join(output_dir, filename), "w") as f:
    for data in data_collection:
        json.dump(data, f)
        f.write('\n')
