import csv, sys
path = sys.argv[1]
total = 0
count = 0
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        v = int(row["value"])
        if v % 3 == 0:
            total += v
        count += 1
print(count, total)
