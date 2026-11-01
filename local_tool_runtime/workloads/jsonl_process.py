import json, sys
path = sys.argv[1]
total = 0
count = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("keep"):
            total += int(obj["value"])
        count += 1
print(count, total)
