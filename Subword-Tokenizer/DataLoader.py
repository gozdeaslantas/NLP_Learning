class DataLoader:

    @staticmethod
    def load_data():
        with open("data/tr_penn-ud-train.txt", encoding="utf-8") as f:
            train_data = f.read()
        with open("data/tr_penn-ud-test.txt", encoding="utf-8") as f:
            test_data = f.read()
        return train_data, test_data

    if __name__ == '__main__':
        load_data()