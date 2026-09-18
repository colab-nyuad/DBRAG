import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import chain
import os
import sqlite3

from openai import OpenAI

from utils import set_seed, read_json

client = OpenAI()

BATCH_SIZE = 2000
def embed(texts, fn=None, hide_progress=False):
    if fn is not None and os.path.isfile(fn):
        return torch.from_numpy(np.load(fn))
    
    embeds = []
    for i in tqdm(range((len(texts) // BATCH_SIZE) + 1), disable=hide_progress):
        _texts = texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        if len(_texts) == 0:
            break
        
        assert len(_texts) >= 1
        
        response = client.embeddings.create(
            model="text-embedding-3-small",  # OpenAI's small embedding model
            input=_texts
        )
        
        vecs = np.array([emb.embedding for emb in response.data])
        embeds.append(vecs)
    
    embeds = np.vstack(embeds)
    
    if fn is not None:
        np.save(fn, embeds)
    
    return torch.from_numpy(embeds)

def decompose_schema(tables):
  r, col_nums = [], []
  for t in tables:
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for t_col in t_cols:
      r.append(f'{t_name}:{t_col}')
    col_nums.append(len(t_cols))
  return r, col_nums

def ravel_t_score(score, col_nums):
  r = []
  idx = 0
  for col_num in col_nums:
    r.append(score[idx:idx+col_num])
    idx += col_num
  assert(idx == len(score))
  return r

def get_sim_scores(dataset, cols_nums=None):
  
  save_fn = f'./data/{dataset}/openai/score1.npy'
  q_embeds_fn = f'../embeddings/spider/query_embeddings_gpt.npy'
  t_embeds_fn = f'./data/{dataset}/openai/t.npy'

  q_embeds, t_embeds = torch.from_numpy(np.load(q_embeds_fn)), torch.from_numpy(np.load(t_embeds_fn))

  print(f'#q, #t: {q_embeds.shape[0]}, {t_embeds.shape[0]}')

  if not os.path.isfile(save_fn):
    sim_scores = []
    for q_embed in tqdm(q_embeds):
      sim_scores.append(F.cosine_similarity(q_embed.unsqueeze(0), t_embeds, dim=1).unsqueeze(0))
    sim_scores = torch.vstack(sim_scores).numpy()
    print(sim_scores.shape)
    
    np.save(save_fn, sim_scores)
  else:
    sim_scores = np.load(save_fn)
  
  return sim_scores

def serialize_table(table):
    db_id, table_name = table['db_id'], table['table_name_original']
    conn1 = sqlite3.connect(f'./data/{dataset}/dev_database/{db_id}/{db_id}.sqlite')
    cursor = conn1.cursor()
    
    # Extract column names from the database
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns = " | ".join([row[1] for row in cursor.fetchall()])
    
    # Fetch first two rows
    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2')
    rows = cursor.fetchall()
    first_2_rows = ["row {}: {}".format(i+1, " | ".join(map(str, row))).lower() for i, row in enumerate(rows)]
    
    conn1.close()
    
    ret_string = f"table name: {table_name.lower()} cols: {columns.lower()} " + " ".join(first_2_rows)

    return ret_string.replace("\n", " ")


# def evaluate(dataset, model, k):
#   corpus_tables = get_corpus(dataset)
#   scores = torch.from_numpy(np.load(f'./data/{dataset}/{model}/score.npy'))
#   top_idxs = scores.topk(k=k, dim=-1)[1]

#   preds = []

#   for top_idx in top_idxs:
#     preds.append([corpus_tables[i] for i in top_idx])
  
#   eval_preds(dataset, preds)

#   write_json(preds, f'./data/{dataset}/{model}/preds_{k}.json')

if __name__ == '__main__':
    set_seed(1234)

    model = ['tapas', 'contriever', 'openai'][2]
    dataset = ['bird', 'spider'][1]

    q_fn, t_fn = f'./data/{dataset}/openai/q.npy', f'./data/{dataset}/openai/t.npy'

    qs = read_json(f'./data/{dataset}/dev.json')
    qs = [q['question'].strip().replace("\n", " ") for q in qs]

    tables = read_json(f'./data/{dataset}/dev_tables.json')

    ts, col_nums = [serialize_table(tables[t]) for t in tables], None

    # embed(qs, q_fn)
    # embed(ts, t_fn)
    get_sim_scores(dataset, cols_nums=col_nums)
    # evaluate(dataset, model, k=20)