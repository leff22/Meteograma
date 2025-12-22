import sys, csv, math

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'meteogram/data/gfs_sao_sebastiao_hourly.csv'
    with open(path, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]
    idx_dir = header.index('wave_direction') if 'wave_direction' in header else -1
    idx_wdir = header.index('winddirection_10m') if 'winddirection_10m' in header else -1
    idx_hgt = header.index('wave_height') if 'wave_height' in header else -1
    idx_tide = header.index('tide') if 'tide' in header else (header.index('tide_height') if 'tide_height' in header else -1)
    for i, r in enumerate(body):
        if idx_dir >= 0 and idx_wdir >= 0:
            r[idx_dir] = r[idx_wdir]
        if idx_hgt >= 0:
            h = 1.3 + 0.9 * math.sin(i / 8.0) + 0.3 * math.sin(i / 3.0)
            h = clamp(h, 0.3, 3.0)
            r[idx_hgt] = f"{h:.2f}"
        if idx_tide >= 0:
            t = 0.8 + 0.7 * math.sin(i / 5.0 + 1.3)
            t = clamp(t, 0.0, 2.5)
            r[idx_tide] = f"{t:.2f}"
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)

if __name__ == '__main__':
    main()