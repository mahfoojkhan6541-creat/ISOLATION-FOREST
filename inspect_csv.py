import csv

with open("data/D1.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("D1 Header:", header[:10])
    row1 = next(reader)
    print("D1 Row 1:", row1[:10])

with open("data/D2.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header2 = next(reader)
    print("D2 Header:", header2[:10])
    row2 = next(reader)
    print("D2 Row 1:", row2[:10])
