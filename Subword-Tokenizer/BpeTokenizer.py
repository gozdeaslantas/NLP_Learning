from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, Lowercase, NFKC
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel
from DataLoader import DataLoader
from PPEvaluator import PPEvaluator
import json


class BytePairEncoder:

    def prepare_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([Lowercase(), NFKC()])
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
        tokenizer.decoder = ByteLevelDecoder()
        return tokenizer

    def prepare_trainer(self):
        return BpeTrainer(vocab_size=30000, show_progress=True, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])


def main():
    tokenizer = BytePairEncoder().prepare_tokenizer()
    trainer = BytePairEncoder().prepare_trainer()
    tokenizer.train(files=["data/tr_penn-ud-train.txt", "data/tr_penn-ud-dev.txt"], trainer=trainer)
    print("Trained vocab size with BPE Tokenizer: {}".format(tokenizer.get_vocab_size()))
    #tokenizer.model.save('data/BPE')

    #tokenizer.model = BPE.from_file('data/BPE/vocab.json', 'data/BPE/merges.txt')
    tokenizer.post_processor = ByteLevel(trim_offsets=True)
    tokenizer.enable_truncation(max_length=512)
    
    encoded = tokenizer.encode('Bu bir kelime oyunudur a dostlar!')
    decoded = tokenizer.decode(encoded.ids)
    print('Decoded: ', decoded)
    
    # Save the vocabulary 
    tokenizer.model.save('result/.')

    with open("result/vocab.json") as f:
       bpe_vocab = json.load(f)
    
    bpe_vocab_list = [(v, k) for k, v in bpe_vocab.items()]
    
    training_corpus, testing_corpus = DataLoader.load_data()
    training_encoding = tokenizer.encode(training_corpus)
    testing_encoding = tokenizer.encode(testing_corpus)

    train_bpe_subwords = training_encoding.tokens
    test_bpe_subwords = testing_encoding.tokens
    
    perplexity = PPEvaluator().compute_pp(2, train_bpe_subwords, test_bpe_subwords)
    print ("Perplexity of BpeTokenizer: {0}".format(perplexity))


if __name__ == '__main__':
    main()
