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
This script runs the covariance computation for ExTrack on a CSV file of track data
using the specified model.
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import logging
import time
import datetime

KEY_LOG_LIKELIHOOD = 'log_likelihood'
KEY_FIT_RESULT = 'result'
KEY_USE_PRECISION = 'use_precision'
KEY_ARGS = 'args'
COL_STATE = 'state'

def process_tracks(path, args):
  # Load previous model
  file_name = os.path.splitext(path)[0]
  file_base = file_name + '.' + str(args.nb_states)
  model = {}
  model_file = f'{file_base}.model.json'
  if not os.path.isfile(model_file):
    if os.path.isfile(os.path.basename(model_file)):
      model_file = os.path.basename(model_file)
    else:
      logging.error(f'No model: {model_file}')
      return
  logging.info(f'Loading model: {model_file}')
  with open(model_file) as f:
    model = json.load(f)

  model_no = args.model
  if model_no == -1:
    model_no = np.argmax(model[KEY_LOG_LIKELIHOOD])
  elif model_no >= len(model[KEY_LOG_LIKELIHOOD]):
    logging.error(f'No model number: {model_no}')
    return
  logging.info(f'Using model number: {model_no}')

  logging.info(f'Reading track data: {path}')

  # Load data
  all_tracks, frames, opt_metrics = extrack.readers.read_table(path,
    # Note: Read all lengths and fit a subset
    lengths=np.arange(args.lengths[0], args.lengths[1] + 1),
    dist_th=args.dist_th,
    # frames_boundaries=[0, 10000], # Ignore to load all frames
    fmt='csv',
    colnames=args.colnames,
    remove_no_disp=True)

  # Optional use of localisation per molecule
  LocErr_type = 1
  input_LocErr = None
  est_precision = None
  if args.col_precision in opt_metrics:
    # Estimate precision
    est_precision = np.median(
      np.concatenate(list(v.flatten() for v in
        opt_metrics[args.col_precision].values())))
    logging.info(f'Estimated localisation precision: {est_precision}')

    if not args.no_precision:
      logging.info('Creating per-localisation precision')
      # Note: The LocError key is ignored if input_LocErr is not None.
      # Leave the error type as 1. This creates parameters with a
      # LocErr key so that the saved model has the same number of keys
      # even if per-localisation precision is used.
      # LocErr_type = None
      input_LocErr = {}
      # opt_metrics stores each column as a dictionary with the same format as
      # the tracks: track length as keys; values as of 2D arrays:
      # dim 0 = track, dim 1 = metric for each time position
      # Here we duplicate the single metric to an XY precision array:
      # dim 0 = track, dim 1 = time position, dim 2 = XY precision
      for k, v in opt_metrics[args.col_precision]:
        input_LocErr[k] = np.repeat(v, 2, axis=1).reshape(v.shape + (2,))

  logging.info('Creating parameters')

  # Create lmfit.parameter.Parameters
  params = extrack.tracking.generate_params(nb_states=args.nb_states,
    LocErr_type=LocErr_type,
    nb_dims=2, # only matters if LocErr_type==2.
    D_max=100, # maximal diffusion coefficient allowed.
    )
  if est_precision is not None:
    # This is clipped to the bounds
    params['LocErr'].value = est_precision

  # Slacken parameter limits to allow x+/-delta for the gradients
  params['LocErr'].min = 0
  for i in range(args.nb_states - 1):
    params[f'D{i}'].min = 0
    params[f'F{i}'].min = 0
    params[f'F{i}'].max = 1
  # Note: F{n-1} is an expression and has no limits set
  params[f'D{args.nb_states - 1}'].min = 0
  for i in range(args.nb_states):
    for j in range(args.nb_states):
      if i != j:
        params[f'p{i}{j}'].min = 0
  params['pBL'].min = 0

  logging.info(f'Initialising to parameters of model number {model_no}')
  for k, v in params.items():
    v.value = model[k][model_no]

  # Fit tracks of limited length
  wanted_keys = [str(k)
    for k in range(args.fit_lengths[0], args.fit_lengths[1] + 1)]
  fit_tracks = dict((k, all_tracks[k]) for k in wanted_keys if k in all_tracks)

  started = time.time()

  # https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult
  uvars = extrack.tracking.param_vars(all_tracks=fit_tracks,
    dt=args.dt, # Time in between frames
    params=params,
    nb_states=args.nb_states, # Number of states
    nb_substeps=args.substeps, # Number of considered transition steps in between consecutive 2 positions.
    cell_dims=[args.cell_dim], # Dimension limits (um)
    frame_len=args.frames, # Number of frames for which the probability is perfectly computed.
    verbose=1 if args.debug else 0,
    workers=args.workers,
    input_LocErr=input_LocErr, # Optional peakwise localization errors used as an input with the same format as all_tracks.
    steady_state=False,
    threshold=args.threshold, # threshold for the fusion of the sequences of states
    max_nb_states=200, # maximum number of sequences of states to consider.
    step=args.step,
    richardson_terms=args.richardson_terms,
    rel_step=args.rel_step,
    num_steps=args.num_steps,
    dd_method=args.dd_method,
    order=args.order,
    maxiter=args.maxiter,
    rtol=args.rtol,
    forward=args.forward,
    )

  rt = time.time() - started
  logging.info(f'Done in {datetime.timedelta(seconds=rt)}')

  if uvars is None:
    logging.info('Failed to compute variances')
    return

  logging.info('Parameters:\n' +
    '\n'.join(f'  {k}={repr(v)}' for k, v in uvars.items()))

  # save model params to a CSV using append.
  # save: name, params+std, run time, options
  data = {
    'name': model_file,
    'step': args.step,
    'rel_step': args.rel_step,
    'num_steps': args.num_steps,
    'dd_method': args.dd_method,
    'order': args.order,
    'maxiter': args.maxiter,
    'rtol': args.rtol,
    'forward': args.forward,
    'time': rt,
  }
  for k, v in uvars.items():
    data[k] = v.n
    data[k + '_std'] = v.s

  df = pd.DataFrame(data, index=[0])

  csv_file = f'{file_base}.vars.csv'
  if os.path.exists(csv_file):
    df2 = pd.read_csv(csv_file)
    df = pd.concat([df2, df])

  df.to_csv(csv_file, index=False)


def parse_args():
  parser = argparse.ArgumentParser(
    description='Program to run compute ExTrack model covariance on CSV track data.')
  parser.add_argument('file', metavar='FILE',
    help='track data file (distances in micrometers)')
  parser.add_argument('--model', type=int, default=-1,
    help='model number (default: %(default)s)')

  parser.add_argument('--workers', dest='workers', type=int,
    default=os.cpu_count(),
    help='number of workers (default: %(default)s)')
  parser.add_argument('--dt', dest='dt', type=float, default=0.01,
    help='frame exposure time (default: %(default)s)')
  parser.add_argument('--nb-states', dest='nb_states', type=int, default=2,
    help='number of states (default: %(default)s)')
  parser.add_argument('--frames', dest='frames', type=int, default=9,
    help='number of frames for which probability is perfectly computed'\
      ' (default: %(default)s)')
  parser.add_argument('--cell-dim', dest='cell_dim', type=float, default=1,
    help='cell dimension limit for the field of view (FOV)'\
      ' (default: %(default)s). A membrane protein in a typical e-coli cell'\
      ' in tirf would have a cell_dims = [0.5,3], in case of cytosolic'\
      ' protein one should imput the depth of the FOV e.g. [0.3] for'\
      ' tirf or [0.8] for hilo.')
  parser.add_argument('--debug', dest='debug',
    action='store_true',
    help='debug output (default: %(default)s)')

  group = parser.add_argument_group('Data')
  group.add_argument('--lengths', dest='lengths', nargs=2, type=int,
    default=[2, 500],
    help='track lengths loaded (default: %(default)s)')
  group.add_argument('--dist-threshold', dest='dist_th', type=float,
    default=0.4,
    help='maximum distance allowed for consecutive positions'
      ' (default: %(default)s)')
  group.add_argument('--no-precision', dest='no_precision',
    action='store_true',
    help='do not use per localisation precision if present'
      ' (default: %(default)s)')
  group.add_argument('--col-names', dest='colnames', nargs=4,
    # Configure for GDSC SMLM csv column headings
    # default=['X', 'Y', 'Time', 'Identifier'],
    # Configure for Spot-On csv column headings
    default=['x', 'y', 't', 'trajectory'],
    help='columns headings for X,Y,T,ID (default: %(default)s)')
  group.add_argument('--col-precision', dest='col_precision',
    # Configure for GDSC SMLM csv column heading
    default='Precision',
    help='columns heading for precision (default: %(default)s)')

  group = parser.add_argument_group('Fitting function')
  group.add_argument('--fit-lengths', dest='fit_lengths', nargs=2, type=int,
    default=[2, 20],
    help='track lengths considered (default: %(default)s)')
  group.add_argument('--threshold', dest='threshold', type=float, default=0.1,
    help='threshold for the fusion of the sequences of states'
      ' (default: %(default)s)')
  group.add_argument('--sub-steps', dest='substeps', type=int, default=1,
    help='number of considered transition steps in between consecutive'\
      ' 2 positions (default: %(default)s)')

  group = parser.add_argument_group('Hessian')
  group.add_argument('--step', type=float,
    default=1e-4,
    help='step (default: %(default)s)')
  group.add_argument('--rel-step',
    action='store_true',
    help='use relative step (default: %(default)s)')
  group.add_argument('--num-steps', type=int,
    default=1,
    help='number of steps (default: %(default)s)')
  group.add_argument('--richardson-terms', type=int,
    default=2,
    help='number of richardson terms used in extrapolation (numdifftools) (default: %(default)s)')
  group.add_argument('--forward',
    action='store_true',
    help='use forward differences; otherwise central (numdifftools) (default: %(default)s)')
  group.add_argument('--dd-method', type=int,
    default=0,
    help='method: 0=numdifftools; 1=scipy hessian (default: %(default)s)')
  group.add_argument('--order', type=int,
    default=2,
    help='order of the finite difference formula to be used (scipy hessian) (default: %(default)s)')
  group.add_argument('--maxiter', type=int,
    default=4,
    help='max iterations (scipy hessian) (default: %(default)s)')
  group.add_argument('--rtol', type=float,
    default=1e-4,
    help='relative tolerance (scipy hessian) (default: %(default)s)')

  args = parser.parse_args()

  # validate
  logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO)

  return args


if __name__ == '__main__':
  args = parse_args()

  # lazy import
  import extrack
  import pandas as pd

  process_tracks(args.file, args)
