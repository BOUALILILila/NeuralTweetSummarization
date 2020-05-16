import sys
from skip_gram_training import Skip_gram
from cbow_training import CBOW
from fasttext_training import FastText

if __name__ == "__main__":
    model=sys.argv[1]
    if model=='cbow':
        m=CBOW()
    elif model=='skip_gram':
        m=Skip_gram()
    elif model=='fasttext':
        m=FastText()
    else:
        print('unkown model')
        sys.exit(0)
    m.train()