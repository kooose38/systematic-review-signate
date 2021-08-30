import  gensim.downloader as api 
from typing import List, Tuple 
import numpy as np 
import pandas as pd 
import time 

class Aurgement():
    def __init__(self):
        self.model = api.load("word2vec-google-news-300")
        self.dataset = []
        
    def _random_text(self, doc: List[str]) -> int:
        '''辞書に登録されていないトークンはスキップする'''
        for _ in doc:
            idx = np.random.randint(0, len(doc))
            word: str = doc[idx]
                if word in self.model:
                    return idx
                else:
                    return -1
        
    def _do_augment(self, docs: List[List[str]]) -> List[List[str]]:
        aug = []
        for doc in docs:
            max_rate = int(len(doc)*0.2)
            
            doc_new: List[str] = doc.copy()
            for _ in range(max_rate):
                idx = self._random_text(doc)
                if idx < 0: break 
                    
                word = doc_new[idx]
                word_similar: Tuple[str, float] = self.model.similar_by_word(word, topn=1)[0]
                doc_new[idx] = word_similar[0]
#                 print(f"{word} -> {word_similar[0]} at {idx}")
            aug.append(doc_new)
        return aug 
    
    def _aug2sentence(aug_doc: List[List[str]]) -> List[str]:
        sentence = []
        for doc in arg_doc:
            sentence.append(" ".join(doc))
        return sentence
    
    def run(dfs: pd.DataFrame) -> pd.DataFrame:
        '''word2vecにより最も近いトークンをランダムで置き換える'''
        assert dfs.columns == ["abstract", "judgement", "abstract_isnan"]
        
        df = dfs.copy()
        df = df[df["judgement"] == 1]
        for doc in df.abstract.to_list():
            data = []
            for word in doc:
                data.append(word)
            self.dataset.append(data)
            
        start = time.time()
        aug1 = self._do_augment(self.dataset)
        sen1 = self._aug2sentence(aug1)
        df_aug1 = pd.DataFrame({"abstract": sen1, "judgement": 1, "abstract_isnan": 0})
        end = time.time()
        print(f"dulation time 1/3 {end-start}")
        
        aug2 = self._do_augment(aug1)
        sen2 = self._aug2sentence(aug2)
        df_aug2 = pd.DataFrame({"abstract": sen2, "judgement": 1, "abstract_isnan": 0})
        end = time.time()
        print(f"dulation time 2/3 {end-start}")
        
        aug3 = self._do_augment(aug2)
        sen3 = self._aug2sentence(aug3)
        df_aug3 = pd.DataFrame({"abstract": sen3, "judgement": 1, "abstract_isnan": 0})
        end = time.time()
        print(f"dulation time 3/3 {end-start}")
        
        df_aug = pd.concat([df_aug1, df_aug2, df_aug3])
        return pd.concat([df, df_aug])
        
            
        
            
                
        