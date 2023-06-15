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
  parser.add_argument('-d', '--delete', dest='delete', nargs='+',
    help='keys to remove from the output')
  parser.add_argument('-i', '--include', dest='include', nargs='+',
    help='keys to include in the output (overrides delete)')

  args = parser.parse_args()
  kd = set(args.delete) if args.delete else set()
  ki = set(args.include) if args.include else set()
  for path in args.file:
    file_name, file_extension = os.path.splitext(path)

    if file_extension == '.json':
      with open(path) as f:
        model = json.load(f)
    else:
      print(f'ERROR: Unknown model for {path}')
      continue;

    print(f'Model: {path}')
    
    if args.best:
      m = np.argmin(model['log_likelihood'])
      for k in model:
        model[k] = model[k][m]

    if ki:
      keys_to_include = ki.intersection(set(model.keys()))
      model = {k: model[k] for k in keys_to_include}
    elif kd:
      keys_to_remove = kd.intersection(set(model.keys()))
      for key in keys_to_remove:
        del model[key]
    
    if args.json:
      print(json.dumps(model, indent=2, sort_keys=True))
    else:
      import pprint
      pp = pprint.PrettyPrinter(indent=2)
      pp.pprint(model)

if __name__ == '__main__':
  main()
