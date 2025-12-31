#!/usr/bin/env python3
"""
PostgreSQL LLM Training Script
Trains a custom transformer model from a text corpus for PostgreSQL-specific tasks.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
from collections import Counter

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token = 0
        self.pad_token = 1
        self.eos_token = 2
        
    def build_vocab(self, corpus_text):
        """Build vocabulary from corpus"""
        # Count characters
        char_counts = Counter(corpus_text)
        
        # Get most common characters
        most_common = char_counts.most_common(self.vocab_size - 3)  # Reserve 3 for special tokens
        
        # Build mappings
        self.char_to_idx = {'<UNK>': self.unk_token, '<PAD>': self.pad_token, '<EOS>': self.eos_token}
        self.idx_to_char = {self.unk_token: '<UNK>', self.pad_token: '<PAD>', self.eos_token: '<EOS>'}
        
        idx = 3
        for char, _ in most_common:
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            idx += 1
        
        # Update vocab_size to actual size
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text):
        """Encode text to token IDs"""
        return [self.char_to_idx.get(char, self.unk_token) for char in text]
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in token_ids if idx != self.pad_token])
    
    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()}
            }, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        tokenizer.vocab_size = len(tokenizer.char_to_idx)
        return tokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PostgreSQLLM(nn.Module):
    """Custom transformer model for PostgreSQL tasks"""
    
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, 
                 dim_feedforward=512, max_seq_length=256, dropout=0.1):
        super(PostgreSQLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.max_seq_length = max_seq_length
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


class TextDataset(Dataset):
    """Dataset for text corpus"""
    
    def __init__(self, corpus_file, tokenizer, max_seq_length=256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into sequences
        sequences = text.split('\n')
        for seq in sequences:
            if len(seq.strip()) > 0:
                tokens = tokenizer.encode(seq)
                if len(tokens) > 0:
                    # Truncate or pad to max_seq_length
                    if len(tokens) > max_seq_length:
                        tokens = tokens[:max_seq_length]
                    else:
                        tokens = tokens + [tokenizer.pad_token] * (max_seq_length - len(tokens))
                    self.samples.append(tokens)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Input is all tokens except last, target is all tokens except first
        input_seq = torch.tensor(tokens[:-1], dtype=torch.long)
        target_seq = torch.tensor(tokens[1:], dtype=torch.long)
        return input_seq, target_seq


def train_model(corpus_file, output_dir, corpus_size_mb, epochs, batch_size, 
                learning_rate, d_model, nhead, num_layers, dim_feedforward,
                max_seq_length, vocab_size):
    """Train the model"""
    
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
    
    print(f"Corpus size: {len(corpus_text)} characters")
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size)
    tokenizer.build_vocab(corpus_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
    print(f"Tokenizer saved to {output_dir}/tokenizer.json")
    
    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(corpus_file, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model
    print("Initializing model...")
    model = PostgreSQLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            output = model(input_seq)
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            target = target_seq.reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'postgres_llm_final.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save config
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'max_seq_length': max_seq_length
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    print("\nTraining completed successfully!")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train PostgreSQL LLM')
    parser.add_argument('--corpus-file', required=True, help='Path to corpus text file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    parser.add_argument('--corpus-size-mb', type=int, default=1, help='Corpus size in MB')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--max-seq-length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    
    args = parser.parse_args()
    
    try:
        train_model(
            args.corpus_file,
            args.output_dir,
            args.corpus_size_mb,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.dim_feedforward,
            args.max_seq_length,
            args.vocab_size
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

