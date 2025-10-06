"""
BERT-based emotion classification model
Fine-tunes pre-trained BERT for multi-emotion detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, NUM_EMOTIONS, EMOTION_LABELS

class EmotionDataset(Dataset):
    """Custom PyTorch dataset for emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
        """
        Initialize emotion dataset
        
        Args:
            texts: Pandas Series or list of text strings
            labels: Pandas Series or list of emotion labels (encoded as integers)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = labels.reset_index(drop=True) if hasattr(labels, 'reset_index') else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertEmotionClassifier:
    """BERT-based emotion classifier"""
    
    def __init__(
        self,
        num_emotions: int = NUM_EMOTIONS,
        model_name: str = BERT_MODEL_NAME
    ):
        """
        Initialize BERT emotion classifier
        
        Args:
            num_emotions: Number of emotion categories
            model_name: Pre-trained BERT model name
        """
        self.num_emotions = num_emotions
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = EMOTION_LABELS
        
        print(f"🔧 Initializing BERT model on device: {self.device}")
        
        # Load pre-trained BERT
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_emotions
        ).to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"✓ BERT model loaded successfully")
    
    def create_data_loaders(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = EmotionDataset(X_train, y_train, self.tokenizer)
        val_dataset = EmotionDataset(X_val, y_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"✓ Data loaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def fine_tune_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune BERT model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        print(f"\n🚀 Starting model training for {epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                batch_count += 1
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Progress update every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_batch_loss = total_loss / batch_count
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Avg Loss: {avg_batch_loss:.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"  Average Training Loss: {avg_loss:.4f}")
            
            # Validation phase
            val_accuracy = self.evaluate_model(val_loader)
            print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        print(f"\n✅ Training completed successfully!")
    
    def evaluate_model(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation/test set
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Accuracy score
        """
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy
    
    def predict_emotion(self, text: str) -> Tuple[str, float]:
        """
        Predict emotion for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (emotion_name, confidence_score)
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class = torch.argmax(predictions, dim=1).cpu().numpy()[0]
        confidence = torch.max(predictions).cpu().numpy()
        
        emotion_name = self.emotion_labels[predicted_class]
        
        return emotion_name, float(confidence)
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict emotions for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tuples (emotion_name, confidence)
        """
        results = []
        for text in texts:
            emotion, confidence = self.predict_emotion(text)
            results.append((emotion, confidence))
        return results
    
    def save_model(self, save_path: str):
        """Save trained model"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save emotion labels mapping
        import json
        labels_path = os.path.join(save_path, 'emotion_labels.json')
        with open(labels_path, 'w') as f:
            json.dump(self.emotion_labels, f)
        
        print(f"✓ Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load emotion labels mapping
        import json
        labels_path = os.path.join(model_path, 'emotion_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.emotion_labels = json.load(f)
        
        print(f"✓ Model loaded from {model_path}")
