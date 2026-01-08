from BPE.tok import Trainer
from BPE.data import batch_iterator

VOCAB_SIZE = 64_000   
BATCH_SIZE = 10_000   # ideal for 24gb ram

def main():
    batch_iter = batch_iterator(BATCH_SIZE)
    Trainer(batch_iter, vocab_size=VOCAB_SIZE)

if __name__ == "__main__":
    main()
