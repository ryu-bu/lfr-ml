import glob, os
import json

os.chdir("train_files")
f = []
for file in glob.glob("*.json"):
    f = open(file,)
    data = json.load(f)

    print(data)

