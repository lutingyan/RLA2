import os

scripts = ["A2C_ce.py", 
           "A2C_c.py",
           "A2C_e.py",
           'A2C.py']

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python3 {script}")  # 如果 Python 3，改成 `python3 {script}`
    print(f"Finished {script}.\n")