from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFKC
from PPEvaluator import PPEvaluator
from DataLoader import DataLoader

class WordpieceTokenizer:        

    def prepare_tokenizer(self):
        tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer=Sequence([Lowercase(), NFKC()])
        return tokenizer
    
    def prepare_trainer(self):
        return WordPieceTrainer(
            vocab_size=30522, show_progress=True, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )


def main():
    tokenizer = WordpieceTokenizer().prepare_tokenizer()
    trainer = WordpieceTokenizer().prepare_trainer()
    tokenizer.train(files=["data/tr_penn-ud-train.txt", "data/tr_penn-ud-dev.txt"], trainer=trainer)

    # Save the vocabulary 
    tokenizer.model.save('result/.')

    with open("result/vocab.txt") as f:
       wp_vocab = f.read()
        
    training_corpus, testing_corpus = DataLoader.load_data()
    training_encoding = tokenizer.encode(training_corpus)
    testing_encoding = tokenizer.encode(testing_corpus)

    train_wp_subwords = training_encoding.tokens
    test_wp_subwords = testing_encoding.tokens
    
    perplexity = PPEvaluator().compute_pp(2, train_wp_subwords, test_wp_subwords)
    print ("Perplexity of Wordpiece: {0}".format(perplexity))

    cc = tokenizer.encode("construction")
    print(cc.tokens)
    print(tokenizer.decode(cc.ids))
if __name__ == '__main__':
    main()