from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import os

def data_iterator(dataset_dir, subset, hf_repos, batch_size=50_000):
    batch = []

    print("📂 Begin reading from Local Dataset...")
    for s in subset:
        path = os.path.join(dataset_dir, s)
        if not os.path.isdir(path):
            continue
            
        print(f"   > Reading: {s}")
        for ds in os.listdir(path):
            file_path = os.path.join(path, ds)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            batch.append(line)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
            except Exception:
                continue

    print("\n🌐 Begin reading from Hugging Face Repos...")
    for repo_name, config, split, col in hf_repos:
        print(f"   > Fetching: {repo_name} ({config})")
        ds = load_dataset(repo_name, config, split=split, streaming=True)
        for idx, batch in enumerate(ds.batch(batch_size=batch_size)):
            if idx > 100_000:
                break

            text = batch.get(col)
            yield text


dataset_dir = "dataset" 
subset = ["math", "art", "astro", "chem", "com-sci", "economics", "general", "linguistic", "medic", "physics", "sci", "social"]

hf_repos = [
    ("pythainlp/thai_food_v1.0", "default", "train", "text"),
    ("pythainlp/thailaw-v1.0", "default", "train", "text"),
    ("pythainlp/thai-wiki-dataset-v3", "default", "train", "text"),
    ("pythainlp/thai-culturax-clean-dataset", "default", "train", "text"),
    ("pythainlp/thai-open-data-go-th", "default", "train", "text"),
    ("pythainlp/prd_news_30112023", "default", "train", "Detail"),
    ("pythainlp/thaigov-corpus", "default", "train", "raw"),
    ("pythainlp/thai-financial-dataset", "default", "train", "text"),
    ("wannaphong/KhanomTanLLM-pretrained-dataset-thai-subset", "default", "train", "text"),
    ("aisingapore/WangchanLION-Curated", "default", "train", "text"),
    ("aisingapore/WangchanLION-Web", "default", "train", "text"),
    ("wikimedia/wikipedia", "20231101.th", "train", "text"),
    ("uonlp/CulturaX", "th", "train", "text"),

    # zh
    ("fjcanyue/wikipedia-zh-cn", "default", "train", "text"),
    ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
    
    # jA
    ("wikimedia/wikipedia", "20231101.ja", "train", "text"),
    
    # en
    ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    ("jtatman/python-code-dataset-500k", "default", "train", "output"),
    ("ajibawa-2023/C-Code-Large", "default", "train", "code"),
]


tokenizer = Tokenizer(BPE(unk_token=None))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=132632, # first size
    special_tokens=["<|BOS|>", "<|PAD|>", "<|EOS|>"]
)

tokenizer.train_from_iterator(
    data_iterator(dataset_dir, subset, hf_repos),
    trainer=trainer
)


tokenizer.post_processor = TemplateProcessing(
    single="<|BOS|> $A <|EOS|>",
    pair="<|BOS|> $A <|EOS|> <|BOS|> $B <|EOS|>",
    special_tokens=[
        ("<|BOS|>", tokenizer.token_to_id("<|BOS|>")),
        ("<|EOS|>", tokenizer.token_to_id("<|EOS|>")),
    ],
)


tokenizer.save("tokenizer.json")
