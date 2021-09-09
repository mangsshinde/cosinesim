import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import joblib

doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"
doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"
doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"
doc_afsan = 'i am an Ai and ml enginner and a instructor in leanrbay'
documents = [doc_trump, doc_election, doc_putin , doc_afsan]

count_vec = CountVectorizer(stop_words='english')
sparse_matrix1 = count_vec.fit_transform(documents)

joblib.dump(sparse_matrix1, "train_matrix.pkl")
joblib.dump(count_vec, "cvector.pkl")
