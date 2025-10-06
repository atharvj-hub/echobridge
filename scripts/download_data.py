"""
Download emotion detection datasets from Hugging Face
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echobridge.utils.logger import setup_logger

logger = setup_logger("download_data")

def download_emotion_dataset():
    """Download emotion dataset from Hugging Face"""
    
    try:
        from datasets import load_dataset
        
        logger.info("📥 Downloading emotion dataset from Hugging Face...")
        
        # Download dataset
        dataset = load_dataset("emotion")
        
        # Create data directory
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits as CSV
        for split in ['train', 'validation', 'test']:
            df = dataset[split].to_pandas()
            
            # Map label IDs to emotion names
            label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
            df['emotion'] = df['label'].map(label_map)
            df = df[['text', 'emotion']]
            
            # Save
            output_path = data_dir / f"emotion_{split}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved {split} set: {len(df)} samples → {output_path}")
        
        logger.info("\n✅ Dataset downloaded successfully!")
        
    except ImportError:
        logger.error("❌ 'datasets' library not installed. Install with: pip install datasets")
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_emotion_dataset()
