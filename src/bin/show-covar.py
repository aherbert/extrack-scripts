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
This script outputs the saved model covariance from ExTrack.
"""

import argparse

def main():
  parser = argparse.ArgumentParser(
    description='Program to show saved ExTrack model covariance.')
  parser.add_argument('file', nargs='+', metavar='FILE', help='model file')

  args = parser.parse_args()

  import pandas as pd

  data = []

  for path in args.file:
    df = pd.read_csv(path)
    data.append(df)

  # Create NA entries if columns are missing from tables, e.g. 2 and 3 state models
  df = pd.concat(data, axis=0).reset_index(drop=True).sort_values(
    'name', ignore_index=True, key=lambda col: col.str.lower())

  print(df.to_csv(index=False).rstrip())

if __name__ == '__main__':
  main()
