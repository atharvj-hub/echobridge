"""Data augmentation techniques for emotion datasets"""

import random
from typing import List

class EmotionDataAugmenter:
    """Augment emotion text data for better model training"""
    
    def __init__(self):
        self.synonyms = {
            'happy': ['joyful', 'delighted', 'pleased', 'glad'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'depressed'],
            'angry': ['furious', 'mad', 'irritated', 'enraged'],
            'scared': ['frightened', 'terrified', 'afraid', 'fearful']
        }
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms"""
        words = text.split()
        for _ in range(n):
            word_to_replace = random.choice(words)
            if word_to_replace.lower() in self.synonyms:
                synonym = random.choice(self.synonyms[word_to_replace.lower()])
                words = [synonym if w == word_to_replace else w for w in words]
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [w for w in words if random.random() > p]
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment_dataset(self, texts: List[str], labels: List[str]) -> tuple:
        """Augment entire dataset"""
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Synonym replacement
            augmented_texts.append(self.synonym_replacement(text))
            augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
