
import os
import json

log_dir = "/home/guch1/logdir/dreamerv3_origin_bankheist_100k/"

print(f"Listing contents of {log_dir}:")
try:
    files = os.listdir(log_dir)
    print(files)
    
    if "metrics.jsonl" in files:
        print("\nFound metrics.jsonl. Reading last few lines:")
        with open(os.path.join(log_dir, "metrics.jsonl"), "r") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(line.strip())
    else:
        print("\nmetrics.jsonl NOT found.")

except FileNotFoundError:
    print(f"Directory not found: {log_dir}")
except PermissionError:
    print(f"Permission denied accessing: {log_dir}")
