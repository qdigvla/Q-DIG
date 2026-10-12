import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from itertools import combinations
import numpy as np

class SimilarityModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.sbert_model = SentenceTransformer(model_name, device=device)
        self.sbert_model.eval()

    def encode(self, prompts):
        return self.sbert_model.encode(prompts)

    def similarity(self, prompts):
        """Compute average pairwise *diversity* (1 - similarity) between
        prompts, despite the method name -- i.e. higher return value means
        the prompts are less similar / more diverse."""
        encoded = self.sbert_model.encode(prompts)

        combs = combinations(encoded, 2)
        all = []
        for i, (a, b) in enumerate(combs):
            sim = self.sbert_model.similarity(a, b)[0][0]
            sim = float(sim)
            all.append(sim)


        avg_similarity = sum(all) / len(all)
        return 1-avg_similarity

    def similarity_to_one(self, prompts, one):
        """Average diversity (1 - similarity) of `prompts` relative to the
        single reference `one` (a string or pre-batched list of strings)."""
        if type(one) == str:  # lowk did this for no reason lol
            one = [one]

        sim = self.sbert_model.similarity(
            self.sbert_model.encode(prompts),
            self.sbert_model.encode(one)
        )

        return 1-float(torch.mean(sim))
    

    def pairwise_bleu(self, prompts):
        smooth = SmoothingFunction().method1
        scores = []
        for i in range(len(prompts)):
            for j in range(len(prompts)):
                if i != j:
                    score = sentence_bleu(
                        [prompts[j].split()],
                        prompts[i].split(),
                        smoothing_function=smooth
                    )
                    scores.append(score)
        return scores


    def bleu_diversity(self, prompts):
        """Compute average pairwise BLEU score between prompts."""
        scores = self.pairwise_bleu(prompts)
        avg_bleu = sum(scores) / len(scores)
        return 1 - avg_bleu