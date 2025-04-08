import os

scripts = ["A3C2.py", "A3C3.py", 
           "A3C4.py"]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python3 {script}")  # 如果 Python 3，改成 `python3 {script}`
    print(f"Finished {script}.\n")