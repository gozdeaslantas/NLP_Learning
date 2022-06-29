from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace


class PPEvaluator:


    def compute_pp(self, n, tokenized_train, tokenized_test): 
        train_data, padded_sents = padded_everygram_pipeline(n, tokenized_train) 
        test_data, padded_sents = padded_everygram_pipeline(n, tokenized_test)  
        model = Laplace(1) 
        model.fit(train_data, padded_sents)

        s = 0
        for i, test in enumerate(test_data):
            p = model.perplexity(test)
            s += p

        perplexity = s/(i+1)
        return perplexity