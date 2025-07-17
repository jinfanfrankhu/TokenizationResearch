from datasets import load_from_disk
from gensim.models import Word2Vec
import os
import multiprocessing

langs = ["ZFinnish", "ZTurkish"]
strategies = ["SUBWORDCORRECTEDBPE5k", "SUBWORDCORRECTEDBPE10k", "SUBWORDCORRECTEDBPE25k", "SUBWORDCORRECTEDBPE50k"]


def make_model(lang, strategy):
    dataset = load_from_disk(rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\{strategy}")
    tokens = dataset['train']['tokens']

    print(f"Training Word2Vec model for {lang} {strategy}...")
    num_cores = multiprocessing.cpu_count()
    model = Word2Vec(sentences=tokens, vector_size=150, window=5, min_count=1, workers=num_cores)
    
    # Save the model
    model_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Word2Vec\{lang}_{strategy}_word2vec.model"
    print(f"attempting to save model to {model_path}")
    model.save(model_path)
    print(f"Model saved to {model_path}")

for lang in langs:
    for strategy in strategies:
        make_model(lang, strategy)



