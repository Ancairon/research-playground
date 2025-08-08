# ramp_cpu.py
import multiprocessing as mp
import time
import math
import random

def burn_cpu():
    """
    Tight loop doing a mix of expensive ops:
      - math.sin/cos calls
      - random numbers
      - large integer multiplications
    """
    x = 1
    while True:
        # mix of FP & int work
        r = random.random()
        _ = math.sin(r) * math.cos(r)  # two trig calls
        x = x * 1234567  # big integer multiply
        x ^= x << 13     # some bit-twiddling
        x ^= x >> 7
        # keep x within some bound
        x = x & ((1 << 64) - 1)


if __name__ == "__main__":
    processes = []
    try:
        for i in range(8):
            p = mp.Process(target=burn_cpu)
            p.start()
            processes.append(p)
            print(f"Started CPU burner #{i+1}")
            time.sleep(10)
        time.sleep(30)
    finally:
        for p in processes:
            p.terminate()
        print("All burners terminated.")
