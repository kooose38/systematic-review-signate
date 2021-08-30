import pandas as pd 
import re 
from typing import List, Dict  
from nltk.stem import WordNetLemmatizer
from nltk import stem 
import nltk

class PreprocessingTABLE():
    def __init__(self):
        self.except_word = [" ", " . ", "the", "a", "of", "and", "in", "to", "with", "for", "was", "were", 
               "is", "that", "this", "as", "we", "i", "I", "there", "they", "you", "", ".", "are", 
               "or", "at", "0", "these", "be", "on", "from", "alt", "has", "many", "ad"]
        self.pos = ["DT", "NN", "JJ", "FW", "NNS", "NNP", "NNPS", "PDT", "PRP", "VBG", "VBD", "VBN", "VBP", "VBZ"]
        nltk.download("all")
        self.stemmer =  stem.PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def _prep_text(self, doc: str) -> str:
        '''クレンジング処理'''
        new_text = []
        for word in doc.strip().split(" "):
            word = re.sub(r" (.*?) ", "", word)
            word = re.sub(r"\u3000", "", word)
            word = word.lower()
            word = word.replace("(", "")
            word = word.replace(")", "")
            word = word.replace(":", "")
            word = word.replace("<", "")
            word = word.replace(">", "")
            word = word.replace(",", "")
            word = word.replace(".", "")
            if word in self.except_word: continue 
            if word == "" : continue 
            new_text.append(word)
        return " ".join(new_text)
    
    def _pos_clean(self, df: pd.DataFrame) -> List[List[str]]:
        '''英語の形態素解析による品詞の抽出'''
        doc_list = []
        for doc in df.abstract.to_list():
            morph = nltk.word_tokenize(doc)
            poses = nltk.pos_tag(morph)
            word_list = []
            for word, pos in poses:
                if pos in self.pos:
                    word = word.lower()
                    word = self.stemmer.stem(word)
                    word_list.append(word)
            doc_list.append(word_list)
        return doc_list
        
        
        
    def transform(self, dfs: pd.DataFrame, is_train: bool=True) -> pd.DataFrame:
        '''データフレーム全体の整形をする'''
        df = dfs.copy()
        df["abstract_isnan"] = df.abstract.isnull().astype(int)
        df["abstract"] = df.abstract.fillna(df.title)
        df.drop(["title", "id"], axis=1, inplace=True)
        
        df["abstract"] = df.abstract.apply(lambda x: self._prep_text(x))
        doc_list = self._pos_clean(df)
        
        df_ = pd.DataFrame([" ".join(w) for w in doc_list])
        df_.columns = ["abstract"]
        df["abstract"] = df_["abstract"]
        
        if is_train:
            '''データ数が多いので削る'''
            df = df.sort_values(by="judgement", ascending=False)[:5600, :]
        return df 