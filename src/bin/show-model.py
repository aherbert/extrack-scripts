#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""
This script outputs the saved model from ExTrack.
"""

import argparse
import numpy as np
import json
import os

def main():
  parser = argparse.ArgumentParser(
    description='Program to show saved ExTrack model parameters.')
  parser.add_argument('file', nargs='+', metavar='FILE', help='model file')

  parser.add_argument('-b', '--best', dest='best', action='store_true',
    help='show the best model')
  parser.add_argument('-j', '--json', dest='json', action='store_true',
    help='json dump')
  parser.add_argument('-c', '--csv', dest='csv', action='store_true',
    help='csv table')
  parser.add_argument('-a', '--all', dest='all', action='store_true',
    help='combined csv table')
  parser.add_argument('-d', '--delete', dest='delete', nargs='+',
    help='keys to remove from the output (overrides include)')
  parser.add_argument('-i', '--include', dest='include', nargs='+',
    help='keys to include in the output')

  args = parser.parse_args()
  kd = set(args.delete) if args.delete else set()
  ki = set(args.include) if args.include else set()

  data = []
  if args.csv:
    import pandas as pd
    kd.add('args')
    kd.add('result')

  for path in args.file:
    file_name, file_extension = os.path.splitext(path)

    if file_extension == '.json':
      with open(path) as f:
        model = json.load(f)
    else:
      print(f'ERROR: Unknown model for {path}')
      continue

    if not args.csv:
      print(f'Model: {path}')

    if args.best:
      m = np.argmax(model['log_likelihood'])
      for k in model:
        model[k] = model[k][m]

    if kd:
      keys_to_remove = kd.intersection(set(model.keys()))
      for key in keys_to_remove:
        del model[key]
    if ki:
      keys_to_include = ki.intersection(set(model.keys()))
      model = {k: model[k] for k in keys_to_include}

    if args.json:
      print(json.dumps(model, indent=2, sort_keys=True))
    elif args.csv:
      # Require an index to use scalar values
      index = [0] if args.best else None
      df = pd.DataFrame(model, index=index).reset_index(drop=True)
      df.insert(loc=0, column='name', value=path)
      if args.all:
        data.append(df)
      else:
        print(df.to_csv(index=False).rstrip())
    else:
      import pprint
      pp = pprint.PrettyPrinter(indent=2)
      pp.pprint(model)

  if data:
    # Create NA entries if columns are missing from tables, e.g. 2 and 3 state models
    df = pd.concat(data, axis=0).reset_index(drop=True).sort_values(
      'name', ignore_index=True, key=lambda col: col.str.lower())
    print(df.to_csv(index=False).rstrip())

if __name__ == '__main__':
  main()
