#!/usr/bin/env python3
import sys
import numpy as np

if len(sys.argv) < 2:
    print('Usage: analyze_tuning_results.py <tuning_results.yaml>')
    sys.exit(1)

path = sys.argv[1]

# The YAML file produced by the tuner may include python-specific tags
# for numpy scalars which yaml.safe_load can't construct. We therefore
# parse the file manually and extract the numeric lists under each
# "errors:" block, preserving order so we can report which result
# index each errors block came from.

all_errors = []
per_result_mean = []
per_result_counts = []
worst_items = []  # tuples (error, result_idx, step_idx)

with open(path, 'r') as f:
    lines = f.readlines()

result_idx = -1
i = 0
N = len(lines)
while i < N:
    line = lines[i]
    # Detect start of a result block: YAML list under 'results:' begins with '-'
    if line.lstrip().startswith('-') and line.startswith('-'):
        # New top-level result entry
        result_idx += 1
        i += 1
        continue

    # Detect an 'errors:' key
    if line.strip().startswith('errors:'):
        # Collect subsequent list items that start with '-' (possibly indented)
        i += 1
        collected = []
        while i < N:
            l = lines[i]
            # Stop if we reach another top-level key in the result (no leading spaces) or a new result '-'
            if l.startswith('-') and l.lstrip().startswith('-'):
                break
            # match list item pattern
            stripped = l.lstrip()
            if stripped.startswith('-'):
                # attempt to parse a float after the hyphen
                try:
                    part = stripped[1:].strip()
                    if part == '':
                        # empty list item, skip
                        i += 1
                        continue
                    val = float(part)
                    collected.append(val)
                except Exception:
                    # Not a simple float - try to extract number via split
                    try:
                        tok = part.split()[0]
                        val = float(tok)
                        collected.append(val)
                    except Exception:
                        pass
                i += 1
                continue
            # If line is blank or less-indented, stop collecting
            if stripped == '' or not l.startswith('  '):
                break
            i += 1

        if collected:
            arr = np.array(collected, dtype=float)
            all_errors.append(arr)
            per_result_mean.append((result_idx, float(np.nanmean(arr))))
            per_result_counts.append((result_idx, int(np.sum(np.isfinite(arr)))))
            finite_idx = np.where(np.isfinite(arr))[0]
            for idx in finite_idx:
                worst_items.append((float(arr[idx]), result_idx, int(idx)))
        continue

    i += 1

if len(all_errors) == 0:
    print('No valid results/errors found in file')
    sys.exit(0)

# concat
concat = np.concatenate(all_errors)
concat = concat[np.isfinite(concat)]

print('Aggregated error stats across all results:')
print('  count:', concat.size)
print('  mean MAPE: %.4f%%' % np.mean(concat))
print('  median MAPE: %.4f%%' % np.median(concat))
print('  75/90/95/99 percentiles: ', ', '.join('%.4f' % p for p in np.percentile(concat, [75,90,95,99])))
print('  count > 10%%: %d' % np.sum(concat > 10.0))
print('  count > 30%%: %d' % np.sum(concat > 30.0))
print('  count > 100%%: %d' % np.sum(concat > 100.0))

# per-result means (sorted)
per_result_mean.sort(key=lambda x: x[1])
print('\nTop 10 best results by mean MAPE:')
for idx, val in per_result_mean[:10]:
    print('  result %d: mean MAPE=%.4f%%' % (idx, val))

print('\nTop 10 worst results by mean MAPE:')
for idx, val in per_result_mean[-10:]:
    print('  result %d: mean MAPE=%.4f%%' % (idx, val))

# worst individual errors across all results
worst_items.sort(reverse=True, key=lambda x: x[0])
print('\nTop 20 individual APEs (error%, result_index, step_index):')
for error, ridx, sidx in worst_items[:20]:
    print('  %.6f %%  | result %d  | step %d' % (error, ridx, sidx))

# Show distribution summary
import math
print('\nDistribution (small buckets):')
for b in [0,1,2,5,10,20,50,100,200,500,1000]:
    if b == 0:
        cnt = np.sum(concat <= 1e-6)
    else:
        cnt = np.sum(concat <= b)
    print('  <=%4s: %6d' % (str(b), int(cnt)))

# Save a small CSV of worst items (first 200) to a temp file for inspection
out_csv = path + '.worst.csv'
with open(out_csv, 'w') as f:
    f.write('error_percent,result_index,step_index\n')
    for error, ridx, sidx in worst_items[:200]:
        f.write('%.6f,%d,%d\n' % (error, ridx, sidx))
print('\nWrote top-200 worst items to', out_csv)
