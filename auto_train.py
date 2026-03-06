import subprocess

print("Collecting data...")
subprocess.run(["python3","auto_collect.py"])

print("Training model...")
subprocess.run(["python3","train.py"])

print("Done! Model ready.")