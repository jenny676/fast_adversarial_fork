import csv

IN_PATH = "/content/drive/MyDrive/FastAT_5%/metrics_old.csv"
OUT_PATH = "/content/drive/MyDrive/FastAT_5%/metrics.csv"
MAX_EPOCH = 15

rows_by_epoch = {}
header = None

with open(IN_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        try:
            epoch = int(row[0])
        except Exception:
            continue

        if epoch <= MAX_EPOCH:
            # keep LAST occurrence of each epoch
            rows_by_epoch[epoch] = row

# write sorted clean file
with open(OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for epoch in sorted(rows_by_epoch.keys()):
        writer.writerow(rows_by_epoch[epoch])

print(f"Wrote clean metrics file: {OUT_PATH}")
print(f"Epochs included: {sorted(rows_by_epoch.keys())}")
