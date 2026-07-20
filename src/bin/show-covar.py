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
  parser.add_argument("-b",
      "--basename",
      dest="basename",
      default=False,
      action=argparse.BooleanOptionalAction,
      help="Use basename for 'name' column (collates filenames from multiple directories) (default: %(default)s)",
  )
  parser.add_argument("-e",
      "--estimates",
      dest="estimates",
      default=False,
      action=argparse.BooleanOptionalAction,
      help="Use results with the error estimates (default: %(default)s)",
  )

  args = parser.parse_args()

  import pandas as pd
  import numpy as np
  import sys
  from os.path import basename

  data = []

  for path in args.file:
    df = pd.read_csv(path)
    if args.basename:
      df['name'] = [basename(x) for x in df['name']]
    if args.estimates:
      df = df[df['F0_std'] != 0]
    if len(df):
      data.append(df)

  # Create NA entries if columns are missing from tables, e.g. 2 and 3 state models
  df = pd.concat(data, axis=0).reset_index(drop=True).sort_values(
    'name', ignore_index=True, key=lambda col: col.str.lower())

  df.drop_duplicates(inplace=True)

  # Fill missing log-likelihood
  df3 = df[pd.notna(df["log_likelihood"])]
  for index, row in df.iterrows():
    if pd.isna(row['log_likelihood']):
      df2 = df3[
        (df3["name"] == row["name"]) &
        (df3["D0"] == row["D0"]) & (df3["F0"] == row["F0"]) &
        (df3["D1"] == row["D1"]) & (df3["F1"] == row["F1"])
      ]
      if len(df2):
        df.loc[index, "log_likelihood"] = df2.iloc[0]["log_likelihood"]

  print(df.to_csv(index=False).rstrip())
  print(f"Datasets: {len(np.unique(df['name']))}", file=sys.stderr)

if __name__ == '__main__':
  main()
