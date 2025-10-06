"""
Generate sample emotion datasets for testing
Creates synthetic emotion data when real datasets are not available
"""

import pandas as pd
import random
from pathlib import Path

# Sample texts for each emotion
EMOTION_SAMPLES = {
    'joy': [
        "I'm so happy about this wonderful news!",
        "This is absolutely amazing and delightful!",
        "I feel great and excited today!",
        "What a beautiful and joyful moment!",
        "I'm thrilled about the results!",
        "This makes me so happy and cheerful!",
        "I love how everything turned out perfectly!",
        "This is the best day ever!",
        "I'm overjoyed with this outcome!",
        "Such wonderful and positive vibes!"
    ],
    'sadness': [
        "I'm feeling really sad and disappointed.",
        "This makes me extremely unhappy.",
        "I feel down and depressed about this.",
        "It's so heartbreaking and sorrowful.",
        "I'm devastated by what happened.",
        "This is such a melancholy situation.",
        "I feel so blue and miserable today.",
        "It's a very gloomy and sad day.",
        "I'm deeply saddened by the news.",
        "This situation is quite depressing."
    ],
    'anger': [
        "I'm absolutely furious about this!",
        "This makes me so angry and mad!",
        "I'm enraged by this behavior!",
        "I can't believe how irritating this is!",
        "This is completely infuriating!",
        "I'm really annoyed and frustrated!",
        "This makes my blood boil!",
        "I'm extremely upset and angry!",
        "This is outrageous and unacceptable!",
        "I'm livid about what happened!"
    ],
    'fear': [
        "I'm really scared about this situation.",
        "This is absolutely terrifying!",
        "I feel so afraid and anxious.",
        "I'm frightened by what might happen.",
        "This is quite alarming and worrying.",
        "I'm nervous and fearful about the outcome.",
        "This gives me anxiety and dread.",
        "I'm terrified of the consequences.",
        "This is very scary and intimidating.",
        "I feel threatened and worried."
    ],
    'surprise': [
        "Wow, I didn't expect that at all!",
        "This is so surprising and unexpected!",
        "I'm absolutely astonished!",
        "What a shocking revelation!",
        "I can't believe this happened!",
        "This is truly amazing and surprising!",
        "I'm stunned by this news!",
        "This caught me completely off guard!",
        "What an unexpected turn of events!",
        "I'm blown away by this surprise!"
    ],
    'disgust': [
        "That's absolutely disgusting and repulsive!",
        "This is so gross and revolting!",
        "I find this extremely distasteful.",
        "This is quite nauseating and unpleasant.",
        "That behavior is disgusting!",
        "This is repugnant and offensive.",
        "I'm disgusted by what I see.",
        "This is vile and sickening.",
        "Such repulsive and foul behavior!",
        "This is absolutely abhorrent!"
    ],
    'neutral': [
        "I'm going to the store later.",
        "The meeting is scheduled for 3 PM.",
        "It's raining outside today.",
        "I need to finish this report.",
        "The project deadline is next week.",
        "We have a team meeting tomorrow.",
        "I'm reading a book right now.",
        "The weather forecast looks okay.",
        "I'll check my email shortly.",
        "The document needs to be reviewed."
    ]
}

def generate_dataset(num_samples_per_emotion=50):
    """Generate synthetic emotion dataset"""
    
    data = []
    
    for emotion, samples in EMOTION_SAMPLES.items():
        for _ in range(num_samples_per_emotion):
            # Randomly pick a base sample
            base_text = random.choice(samples)
            
            # Add slight variations
            text = base_text
            if random.random() > 0.5:
                # Add punctuation variations
                text = text.rstrip('!.?') + random.choice(['.', '!', '...'])
            
            data.append({
                'text': text,
                'emotion': emotion
            })
    
    # Shuffle data
    random.shuffle(data)
    
    return pd.DataFrame(data)

def main():
    """Generate and save sample datasets"""
    
    print("🎲 Generating sample emotion datasets...")
    
    # Create data directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    train_df = generate_dataset(num_samples_per_emotion=50)
