import os
import time


rounds=900

for i in range(0,rounds):
    os.system(f"python3 base.py {i}")
    time.sleep(1)

