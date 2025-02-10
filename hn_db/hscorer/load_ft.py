import fasttext


def load_ft(path: str = "data/ft/cc.fr.300.bin") -> fasttext.FastText._FastText:
    model = fasttext.load_model(path)
    return model


model = load_ft()
