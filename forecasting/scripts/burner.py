# ramp_cpu.py
import multiprocessing as mp
import time
import signal

# A simple busy‐loop


def burn_cpu():
    x = 0
    while True:
        x += 1


if __name__ == "__main__":
    processes = []
    try:
        for i in range(8):
            p = mp.Process(target=burn_cpu)
            p.start()
            processes.append(p)
            print(f"Started CPU burner #{i+1}")
            time.sleep(30)
        time.sleep(30)
    finally:
        for p in processes:
            p.terminate()
        print("All burners terminated.")
