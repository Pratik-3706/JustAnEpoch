import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import download_and_save_dataset

download_and_save_dataset(subset_size=5)
