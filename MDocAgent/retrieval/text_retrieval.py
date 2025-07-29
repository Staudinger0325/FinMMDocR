import os
import json
import sys
import time
from tqdm import tqdm
from ragatouille import RAGPretrainedModel

log_dir = ""
os.makedirs(log_dir, exist_ok=True)
log_time = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"text_retrieval_{log_time}.log")
sys.stdout = open(log_file, "a", encoding="utf-8")
sys.stderr = sys.stdout

from retrieval.base_retrieval import BaseRetrieval
from mydatasets.base_dataset import BaseDataset

class ColbertRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
    
    def prepare(self, dataset: BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        RAG = RAGPretrainedModel.from_pretrained("")
        doc_index:dict = {}
        error = 0
        total_start = time.time()
        for sample_idx, sample in enumerate(tqdm(samples)):
            if self.config.r_text_index_key in sample and os.path.exists(sample[self.config.r_text_index_key]):
                print(f"[Skip] {sample.get('doc_id', sample_idx)} skipped")
                continue
            if sample[self.config.doc_key] in doc_index:
                sample[self.config.r_text_index_key] = doc_index[sample[self.config.doc_key]]
                print(f"[Skip] {sample.get('doc_id', sample_idx)} skipped")
                continue
            print(f"[Doc] start processing doc: {sample.get('doc_id', sample_idx)}")
            doc_start = time.time()
            content_list = dataset.load_processed_content(sample)
            text = [content.txt.replace("\n", "") for content in content_list]
            page_times = []
            try:
                def page_index_hook(page_texts):
                    page_time_stamps = []
                    for i, t in enumerate(page_texts):
                        t0 = time.time()
                        _ = RAG.encode([t])
                        t1 = time.time()
                        page_time_stamps.append(t1-t0)
                        print(f"[Page] doc{sample.get('doc_id', sample_idx)} page{i+1}/{len(page_texts)} time usage: {t1-t0:.3f}s")
                    return page_time_stamps
                page_times = page_index_hook(text)
                index_path = RAG.index(index_name=dataset.config.name+ "-" + self.config.text_question_key + "-" + sample[self.config.doc_key], collection=text)
                doc_index[sample[self.config.doc_key]] = index_path
                sample[self.config.r_text_index_key] = index_path
            except Exception as e:
                error += 1
                if error>len(samples)/100:
                    print("Too many error cases. Exit process.")
                    import sys
                    sys.exit(1)
                print(f"Error processing {sample[self.config.doc_key]}: {e}")
                sample[self.config.r_text_index_key] = ""
            doc_end = time.time()
            print(f"[Doc] doc{sample.get('doc_id', sample_idx)} total index time usage: {doc_end-doc_start:.3f}s")
        total_end = time.time()
        print(f"[ALL] total doc index time usage: {total_end-total_start:.3f}s")
        dataset.dump_data(samples, use_retreival = True)
        return samples

    def find_sample_top_k(self, sample, top_k: int, page_id_key: str):
        if not os.path.exists(sample[self.config.r_text_index_key]+"/pid_docid_map.json"):
            print(f"Index not found for {sample[self.config.r_text_index_key]}/pid_docid_map.json.")
            return [], []
        with open(sample[self.config.r_text_index_key]+"/pid_docid_map.json",'r') as f:
            pid_map_data = json.load(f)
        unique_values = list(dict.fromkeys(pid_map_data.values()))
        value_to_rank = {val: idx for idx, val in enumerate(unique_values)}
        pid_map = {int(key): value_to_rank[value] for key, value in pid_map_data.items()}
        
        query = sample[self.config.text_question_key]
        # 加载searcher计时
        t0 = time.time()
        RAG = RAGPretrainedModel.from_index(sample[self.config.r_text_index_key])
        t1 = time.time()
        print(f"[Time] loading searcher time usage: {t1-t0:.3f}s")
        # 检索计时
        t2 = time.time()
        results = RAG.search(query, k=len(pid_map))
        t3 = time.time()
        print(f"[Time] retireval usage: {t3-t2:.3f}s")
        
        top_page_indices = [pid_map[page['passage_id']] for page in results]
        top_page_scores = [page['score'] for page in results]
        
        if page_id_key in sample:
            page_id_list = sample[page_id_key]
            assert isinstance(page_id_list, list)
            filtered_indices = []
            filtered_scores = []
            for idx, score in zip(top_page_indices, top_page_scores):
                if idx in page_id_list:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)
            return filtered_indices[:top_k], filtered_scores[:top_k]
        
        return top_page_indices[:top_k], top_page_scores[:top_k]
        
    def find_top_k(self, dataset: BaseDataset, force_prepare=False):
        top_k = self.config.top_k
        samples = dataset.load_data(use_retreival=True)
        
        if self.config.r_text_index_key not in samples[0] or force_prepare:
            samples = self.prepare(dataset)
                
        for sample in tqdm(samples):
            top_page_indices, top_page_scores = self.find_sample_top_k(sample, top_k=top_k, page_id_key = dataset.config.page_id_key)
            sample[self.config.r_text_key] = top_page_indices
            sample[self.config.r_text_key+"_score"] = top_page_scores
        path = dataset.dump_data(samples, use_retreival=True)
        print(f"Save retrieval results at {path}.")