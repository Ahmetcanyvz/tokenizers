from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer

def get_vocab_list(tok):
    # token -> id dict
    v = tok.get_vocab()
    # invert & order by id for readability
    inv = {i:t for t,i in v.items()}
    return [inv[i] for i in sorted(inv)]

def run_step(corpus, seed_vocab, target_size):
    tok = Tokenizer(Unigram())  # model will be overwritten by training
    trainer = CompressionTrainer(
        vocab_size=target_size,
        show_progress=False,
        max_piece_length=8,
        seed_size=10000,
        seed_vocab=seed_vocab,      # <-- use our exact vocab
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok

# Controlled corpus (no spaces in spans)
corpus = ["abcd", "abcd", "ab"]
seed_vocab = ["a","b","c","d","ab","abcd"]   # Σ + extras

print("Initial seed vocab:", seed_vocab)

# Step 0: target = initial size (no prune)
tok0 = run_step(corpus, seed_vocab, target_size=len(seed_vocab))
v0 = get_vocab_list(tok0)
print("Step 0 vocab:", v0)
print("Step 0 segmentations:", [tok0.encode(s).tokens for s in corpus])

# Step 1: prune one → target = len-1
tok1 = run_step(corpus, seed_vocab, target_size=len(seed_vocab)-1)
v1 = get_vocab_list(tok1)
removed1 = sorted(set(v0) - set(v1))
print("\nStep 1 removed:", removed1)
print("Step 1 vocab:", v1)
print("Step 1 segmentations:", [tok1.encode(s).tokens for s in corpus])

# Step 2: prune one more → target = len-2
tok2 = run_step(corpus, v1, target_size=len(v1)-1)  # use previous vocab as next seed
v2 = get_vocab_list(tok2)
removed2 = sorted(set(v1) - set(v2))
print("\nStep 2 removed:", removed2)
print("Step 2 vocab:", v2)
print("Step 2 segmentations:", [tok2.encode(s).tokens for s in corpus])