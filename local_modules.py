import sys
from pathlib import Path

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

#print("Updated path:", sys.path)