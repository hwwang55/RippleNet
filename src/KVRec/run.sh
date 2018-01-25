#!/bin/bash

python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 64 --reg_kg 0.1 --kg_ratio 1.0
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 128 --reg_kg 0.1 --kg_ratio 1.0

python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.01 --kg_ratio 1.0
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.05 --kg_ratio 1.0
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 1.0
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.5 --kg_ratio 1.0
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 1.0 --kg_ratio 1.0

python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.2
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.4
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.6
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.8
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 1.0

python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.01 --kg_ratio 1.0
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.05 --kg_ratio 1.0
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 1.0
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.5 --kg_ratio 1.0
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 1.0 --kg_ratio 1.0

python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.2
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.4
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.6
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.8
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 1.0

python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.1
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.3
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.5
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.7
python main.py --task movie  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.9

python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.1
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.3
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.5
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.7
python main.py --task book  --n_user_click_topk 0 --n_sample_topk 3000  --n_hops 2 --n_entity_emb 50 --reg_kg 0.1 --kg_ratio 0.9



