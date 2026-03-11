import os
import tempfile
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers.trainers import ParityBpeTrainer

train_parquets = [
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/arb_Arab.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/cmn_Hani.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/hin_Deva.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/jpn_Jpan.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/kor_Hang.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/rus_Cyrl.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/tha_Thai.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/ben_Beng.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/tam_Taml.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/kat_Geor.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/amh_Ethi.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/khm_Khmr.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/mya_Mymr.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/heb_Hebr.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/mal_Mlym.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/lao_Laoo.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/swh_Latn.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/fin_Latn.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/yor_Latn.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb2/eus_Latn.parquet",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset/fineweb/CC-MAIN-2013-20.parquet",
]

dev_files = [
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/arb_Arab.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/cmn_Hani.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/hin_Deva.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/jpn_Jpan.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/kor_Hang.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/rus_Cyrl.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/tha_Thai.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/ben_Beng.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/tam_Taml.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/kat_Geor.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/amh_Ethi.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/khm_Khmr.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/mya_Mymr.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/heb_Hebr.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/mal_Mlym.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/lao_Laoo.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/swh_Latn.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/fin_Latn.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/yor_Latn.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/eus_Latn.txt",
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data/eng_Latn.txt",
]


def parquet_to_text(parquet_path, output_dir, text_column="text", batch_size=4096):
    """Convert a parquet file to a UTF-8 text file (one document per line)."""
    basename = os.path.splitext(os.path.basename(parquet_path))[0] + ".txt"
    out_path = os.path.join(output_dir, basename)
    pf = pq.ParquetFile(parquet_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
            for value in batch.column(text_column):
                text = value.as_py()
                if text:
                    # Replace newlines within a document so each line = one document
                    f.write(text.replace("\n", " ") + "\n")
    print(f"  Converted {parquet_path} -> {out_path}")
    return out_path


# Convert all parquet files to text
text_dir = tempfile.mkdtemp(prefix="pa_bpe_text_")
print(f"Converting parquet files to text in {text_dir} ...")
train_files = [parquet_to_text(p, text_dir) for p in train_parquets]

# Set up tokenizer with pre-tokenizer (user-configurable)
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),
    pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=False),
])

# Train
trainer = ParityBpeTrainer(num_merges=128000, variant="base")
trainer.train(tokenizer, train_files=train_files, dev_files=dev_files)

# Test
encoded = tokenizer.encode("Hello world")
print(encoded.tokens)

# Save / load
tokenizer.save("test_pabpe_tok.json")

tokenizer = Tokenizer.from_file("test_pabpe_tok.json")
