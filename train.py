import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BiDataset, causal_mask
from model import build_transformer, Transformers
from config import get_weights_file_path, get_config
import tqdm

def get_all_senetences(ds, lang):
    for example in ds:
        yield example['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_tokens = "<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_senetences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    dataset = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    #Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, dataset, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset, config['lang_tgt'])

    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])
    train_dataset = BiDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_dataset = BiDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_src_len = 0
    max_tgt_len = 0

    for example in dataset:
        src_ids = tokenizer_src.encode(example['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(example['translation'][config["lang_tgt"]]).ids

        max_src_len = max(max_src_len, len(src_ids))
        max_tgt_len = max(max_tgt_len, len(tgt_ids))
    
    print(f"Max src len: {max_src_len}")
    print(f"Max tgt len: {max_tgt_len}")

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle= True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["dmodel"])
    return model

def train_model(config):
    #Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm.tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) #(batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) #(batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) #(batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) #(batch, seq_len, dmodel)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(batch, seq_len, dmodel)
            proj_output = model.project(decoder_output) #(batch, seq_len, vocab_size)

            labels = batch["labels"].to(device) #(batch, seq_len) --> (batch*seq_len)
            # prj_output: (batch, seq_len, vocab_size) --> (batch*seq_len, vocab_size)
            loss = criterion(proj_output.view(-1, proj_output.size(-1)), labels.view(-1))
            batch_iterator.set_postfix(f"loss : {loss.item():6.3f}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            
            #Backpropagation
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        #Save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        torch.save({
            "epoch": epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == "__main__":
    config = get_config()
    train_model(config)