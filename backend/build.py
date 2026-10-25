import random
import time
from pathlib import Path

import numpy as np
from services.extract import get_embedding

DATA = Path("data/for-2seconds")
CACHE = Path("cache")
LABELS = {"real": 0, "fake": 1}      
MAX = {"training": 500, "validation": 100, "testing": 100}

#builds list of file,label pairs
def list_files(section):
    items = []
    for name, label in LABELS.items():
        files = sorted((DATA / section / name).glob("*.wav"))
        limit = (MAX or {}).get(section)
        if limit and len(files) > limit:
            files = random.Random(0).sample(files, limit)  
        items += [(f, label) for f in files]
    #shuffle data (real/fase) together
    random.Random(0).shuffle(items)
    return items

#gets the list and caches an extraction of each pair in it
def build(section):
    items = list_files(section)
    print(f"{section}: {len(items)} clips")
    x, y = [], []
    start = time.perf_counter()
    for i, (path, label) in enumerate(items, 1):
        x.append(get_embedding(str(path)))
        y.append(label)
        #estimated time remaining counter
        if i % 50 == 0 or i == len(items):
            rate = (time.perf_counter() - start) / i
            print(f"  {i}/{len(items)}  {rate:.2f}s/clip  ~{rate * (len(items) - i) / 60:.1f} min left")
    CACHE.mkdir(exist_ok=True)
    #saves in /cache
    np.save(CACHE / f"X_{section}.npy", np.stack(x).astype(np.float32))
    np.save(CACHE / f"y_{section}.npy", np.array(y, dtype=np.int64))

#for solo run in cli
if __name__ == "__main__":
    for section in ("validation", "testing", "training"):
        build(section)