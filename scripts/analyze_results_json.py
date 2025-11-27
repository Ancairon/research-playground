#!/usr/bin/env python3
import sys, json
import numpy as np

if len(sys.argv) < 2:
    print('Usage: analyze_results_json.py <results.json>')
    sys.exit(1)

path = sys.argv[1]
with open(path, 'r') as f:
    data = json.load(f)

preds = data.get('predictions', [])
errs = []
pairs = []
for i, p in enumerate(preds):
    a = p.get('actual')
    pred = p.get('value')
    if a is None or pred is None:
        continue
    den = (abs(a) + abs(pred)) / 2.0
    if den < 1e-6:
        continue
    ape = abs(a - pred) / den * 100.0
    ape = min(ape, 1000.0)
    errs.append(ape)
    pairs.append((i, p.get('timestamp'), a, pred, ape))

arr = np.array(errs, dtype=float)
if arr.size == 0:
    print('No valid errors found')
    sys.exit(0)

print('Count:', arr.size)
print('Mean MAPE (calc): %.6f%%' % np.mean(arr))
print('Median MAPE: %.6f%%' % np.median(arr))
print('Std: %.6f' % np.std(arr))
print('Min: %.6f  Max: %.6f' % (np.min(arr), np.max(arr)))
print('Percentiles 75/90/95/99:', ','.join('%.6f' % v for v in np.percentile(arr, [75,90,95,99])))
print('Count > 10%%:', int(np.sum(arr>10.0)))
print('Count > 30%%:', int(np.sum(arr>30.0)))
print('Count > 100%%:', int(np.sum(arr>100.0)))

# show top offending steps
sorted_idx = np.argsort(-arr)
print('\nTop 20 individual APEs (error%, index, timestamp, actual, pred):')
for idx in sorted_idx[:20]:
    i, ts, a, pred, ape = pairs[idx]
    print('  %.6f%% | idx=%d | %s | actual=%.6f | pred=%.6f' % (ape, i, ts, a, pred))

# Show a small histogram buckets
print('\nBuckets: <=1, <=5, <=10, <=20, <=50, >50')
print(' <=1: %d' % int(np.sum(arr<=1.0)))
print(' <=5: %d' % int(np.sum(arr<=5.0)))
print(' <=10: %d' % int(np.sum(arr<=10.0)))
print(' <=20: %d' % int(np.sum(arr<=20.0)))
print(' <=50: %d' % int(np.sum(arr<=50.0)))
print(' >50: %d' % int(np.sum(arr>50.0)))

# Save worst 200 to CSV
out_csv = path + '.worst.csv'
with open(out_csv, 'w') as f:
    f.write('error_percent,index,timestamp,actual,pred\n')
    for idx in sorted_idx[:200]:
        i, ts, a, pred, ape = pairs[idx]
        f.write('%.6f,%d,%s,%.6f,%.6f\n' % (ape, i, ts, a, pred))
print('\nWrote worst-200 to', out_csv)
