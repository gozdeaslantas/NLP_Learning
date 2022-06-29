from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.normalizers import Sequence, Lowercase, NFKC
from DataLoader import DataLoader
from PPEvaluator import PPEvaluator
import json

class UnigramLM:        

    def prepare_tokenizer(self):
        tokenizer = Tokenizer(Unigram())
        tokenizer.normalizer=Sequence([Lowercase(), NFKC()])
        return tokenizer
    
    def prepare_trainer(self):
        return UnigramTrainer(vocab_size=30000, show_progress=True, min_frequency=2, unk_id=2, special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"])


def main():
    tokenizer = UnigramLM().prepare_tokenizer()
    trainer = UnigramLM().prepare_trainer()
    tokenizer.train(files=["data/tr_penn-ud-train.txt", "data/tr_penn-ud-dev.txt"], trainer=trainer)
    
    encoded = tokenizer.encode('Bu bir kelime oyunudur a dostlar!')
    decoded = tokenizer.decode(encoded.ids)
    print('Decoded: ', decoded)
    
    print("Trained vocab size with UNIGRAM Tokenizer: {}".format(tokenizer.get_vocab_size()))
    
    # Save the vocabulary 
    tokenizer.model.save('result/.')

    with open("result/unigram.json") as f:
       unigram_vocab = json.load(f)
    
    unigram_vocab_list = [(v, k) for k, v in unigram_vocab.items()]
    
    training_corpus, testing_corpus = DataLoader.load_data()
    training_encoding = tokenizer.encode(training_corpus)
    testing_encoding = tokenizer.encode(testing_corpus)

    train_unigram_subwords = training_encoding.tokens
    test_unigram_subwords = testing_encoding.tokens
    
    perplexity = PPEvaluator().compute_pp(2, train_unigram_subwords, test_unigram_subwords)
    print ("Perplexity of UnigramLM: {0}".format(perplexity))
if __name__ == '__main__':
    main()