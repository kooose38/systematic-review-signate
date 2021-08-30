from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd 
from typing import Union , List
import plotly.express as px 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class TfIdf():
    def __init__(self):
        self.model = TfidfVectorizer(lowercase=False, tokenizer=self._tokenizer_identily)
        
    def _tokenizer_identily(self, x):
        return x 

    def _create_seq(self, df) -> List[List[str]]:
        seq = []
        for d in df.abstract.to_list():
            word = []
            for w in d.split(" "):
                word.append(w)
            seq.append(word)
        return seq 

    def fit(self, x_train: pd.DataFrame):
        seq = self._create_seq(x_train)
        self.model.fit(seq)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        seq = self._create_seq(df)
        return  self.model.transform(seq).toarray()
    

class PCA_:
    def __init__(self):
        self.model = None 

    def fit(self, x_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame], 
          n_components: int=2, is_plot: bool=False) -> np.ndarray:
        self.model = PCA(n_components=n_components).fit(x_train)

        if is_plot and n_components == 2:
            self._plot(x_train, y_train)

        return self.model.explained_variance_ratio_

    def _plot(self, x_train, y_train):
        x_pca = self.model.fit_transform(x_train)
        fig = px.scatter(x=x_pca[:, 0], y=x_pca[:, 1], color=y_train, title="PCA:")
        fig.show()

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.transform(x)