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
This script runs ExTrack on a CSV file of track data, saves the model and
outputs the predicted states for the track.
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import logging
import time

KEY_LOG_LIKELIHOOD = 'log_likelihood'
KEY_USE_PRECISION = 'use_precision'
KEY_ARGS = 'args'
COL_STATE = 'state'

def process_tracks(path, args):
  # lazy import
  import extrack
  logging.info(f'Reading track data: {path}')

  # Read all additional columns to allow original data to be saved
  orig_colnames = []
  opt_colnames = []
  with open(path) as f:
    for h in f.readline().strip('\n').split(','):
      if h:
        orig_colnames.append(h)
        if h not in args.colnames:
          opt_colnames.append(h)

  # Load data
  all_tracks, frames, opt_metrics = extrack.readers.read_table(path,
    # Note: Read all lengths and fit a subset                                                               
    lengths=np.arange(args.lengths[0], args.lengths[1] + 1),
    dist_th=args.dist_th,
    # frames_boundaries=[0, 10000], # Ignore to load all frames
    fmt='csv',
    colnames=args.colnames,
    remove_no_disp=True,
    opt_colnames=opt_colnames) # Name of the optional metrics to catch

  # Load previous model
  # This is only valid for the same number of states.
  file_name = os.path.splitext(path)[0]
  file_base = file_name + '.' + str(args.nb_states)
  model = {}
  model_file = f'{file_base}.model.json'
  if os.path.isfile(model_file):
    logging.info(f'Loading previous model: {model_file}')
    with open(model_file) as f:
      model = json.load(f)

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
      logging.info(f'Creating per-localisation precision')
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

  logging.info(f'Creating parameters')

  # Create lmfit.parameter.Parameters
  params = extrack.tracking.generate_params(nb_states=args.nb_states,
    LocErr_type=LocErr_type,
    nb_dims=2, # only matters if LocErr_type==2.
    LocErr_bounds=args.loc_bounds, # the initial guess on LocErr will be the geometric mean of the boundaries.
    D_max=args.d_max, # maximal diffusion coefficient allowed.
    Fractions_bounds=args.f_bounds,
    # estimated_LocErr=[0.022],
    estimated_Ds=args.estimated_d, # D will be arbitrary spaced from 0 to D_max if None, otherwise input a list of Ds for each state
    estimated_Fs=args.estimated_f, # fractions will be equal if None, otherwise input a list of fractions for each state
    estimated_transition_rates=args.estimated_rate, # transition rate per step. example [0.1,0.05,0.03,0.07,0.2,0.2] for a 3-state model.
    )
  if est_precision is not None:
    # This is clipped to the bounds
    params['LocErr'].value = est_precision

  # Initialise model with defaults
  for param in params:
    model.setdefault(param, [])
  for param in [KEY_LOG_LIKELIHOOD, KEY_USE_PRECISION, KEY_ARGS]:
    model.setdefault(param, [])

  # Fit tracks of limited length
  wanted_keys = [str(k)
    for k in range(args.fit_lengths[0], args.fit_lengths[1] + 1)]
  fit_tracks = dict((k, all_tracks[k]) for k in wanted_keys if k in all_tracks)

  # Working arguments
  args2 = vars(args).copy()
  del args2['file']

  for i in range(args.repeats):
    logging.info(f'Fitting {i+1}/{args.repeats}')
    start_time = time.time()
    # https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult
    model_fit = extrack.tracking.param_fitting(all_tracks=fit_tracks,
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
      method='bfgs')
    t = time.time() - start_time

    if args.debug and not model_fit.success:
      logging.info(f'Failed: {model_fit.message}')

    ll = -model_fit.residual[0]
    # Information criterion
    # n = number of ??? (localisations, displacements, ...)
    # k = number of fit parameters
    n = sum(len(b) for a in all_tracks.values() for b in a)
    k = model_fit.nvarys
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    logging.info(f'LL {ll} : AIC {aic} : BIC {bic} (in {t} seconds)')

    # Do not overwrite initial params
    params2 = model_fit.params
    logging.info('Parameters:\n' +
      '\n'.join(f'  {k}={v.value}' for k, v in params2.items()))

    for param in params2:
      model[param].append(params2[param].value)
    model[KEY_LOG_LIKELIHOOD].append(ll)
    model[KEY_USE_PRECISION].append(input_LocErr is not None)
    model[KEY_ARGS].append(args2)

  # Save model
  if args.repeats > 0:
    logging.info(f'Saving new model: {model_file}')
    with open(model_file, 'w') as f:
      json.dump(model, f, indent=2, sort_keys=True)

  if not model.get(KEY_LOG_LIKELIHOOD):
    logging.warning('Missing fitted model, skipping prediction')
    return

  # Extract best model for prediction
  m = np.argmax(model[KEY_LOG_LIKELIHOOD])
  for k, v in params.items():
    v.value = model[k][m]
  if not model[KEY_USE_PRECISION][m]:
    # No per-localisation error used for the best model
    input_LocErr = None
  substeps = model['args'][m]['substeps']

  # optional position refinement
  if args.refine:
    logging.info('Generating position refinement')
    LocErr, ds, Fs, TrMat, pBL = extrack.tracking.extract_params(params, 
      dt=args.dt,
      nb_states=args.nb_states,
      nb_substeps=substeps,
      input_LocErr=input_LocErr)

    # Note: No example provided of using per-localisation error during
    # position refinement. It is unknown if this will work when using precision.
    LocErr = LocErr[0][0,0,0]
    #print(LocErr)
    all_mus, all_refined_sigmas = extrack.refined_localization.position_refinement(all_tracks,
      LocErr, ds, Fs, TrMat, frame_len=args.frames)

    refined_pos0 = {}
    refined_pos1 = {}
    refined_LocErrs = {}
    for l in all_tracks.keys():
      refined_pos0[l] = all_mus[l][:,:,0]
      refined_pos1[l] = all_mus[l][:,:,1]
      refined_LocErrs[l] = all_refined_sigmas[l][:,:] # this corresponds to a shared sigma for x and y axes.

    # Add to optional metric data
    opt_metrics['refined_x_pos'] = refined_pos0
    opt_metrics['refined_y_pos'] = refined_pos1
    opt_metrics['refined_LocErrs'] = refined_LocErrs
  
  # optional state duration histogram
  if args.durations:
    from extrack.histograms import len_hist
    logging.info('Generating state duration histogram')
    LocErr, ds, Fs, TrMat, pBL = extrack.tracking.extract_params(params, 
      dt=args.dt,
      nb_states=args.nb_states,
      nb_substeps=substeps,
      input_LocErr=input_LocErr)

    len_hists = len_hist(all_tracks, params, args.dt,
      cell_dims=[args.cell_dim],
      nb_states=args.nb_states,
      workers=args.workers,
      nb_substeps=substeps,
      max_nb_states=500,
      input_LocErr=input_LocErr)
     
    # Save to CSV
    DATA = None
    for k, hist in enumerate(len_hists.T):
      df = pd.DataFrame({str(k): hist/np.sum(hist)})
      if DATA is None:
        DATA = df
      else:
        DATA = pd.concat([DATA, df], axis=1)
    DATA = DATA.fillna(0)
    DATA['seconds'] = pd.Series(np.arange(1,DATA.shape[0]+1)*args.dt)
    hist_file = file_base + '.hist.csv'
    DATA.to_csv(hist_file, index=False,
      columns=['seconds']+list(str(a) for a in range(args.nb_states)))
    logging.info(f'Saved state duration histogram: {hist_file}')

  # Predict
  logging.info('Generating predictions')
  pred_Bs = extrack.tracking.predict_Bs(all_tracks,
    dt=args.dt,
    # lmfit parameters used for the model
    params=params,
    cell_dims=[args.cell_dim],
    nb_states=args.nb_states,
    frame_len=args.frames,
    workers=args.workers,
    input_LocErr=input_LocErr)

  # Convert to data frame
  DATA = extrack.exporters.extrack_2_pandas(all_tracks, pred_Bs,
    frames=frames, opt_metrics=opt_metrics)
  # Add category column based on max probability state
  col = []
  for k in pred_Bs:
    col.append(np.argmax(pred_Bs[k], axis=2).flatten())
  DATA[COL_STATE] = np.concatenate(col)
  # Rename the X,Y,T,ID columns from the ExTrack names to the original names
  DATA.rename(columns=dict(zip(['X','Y','frame','track_ID'], args.colnames)),
    inplace=True)
  # Re-order to the original
  cols = orig_colnames + [COL_STATE] +\
    list(f'pred_{i}' for i in range(args.nb_states))
  if args.refine:
    cols = cols + ['refined_x_pos', 'refined_y_pos', 'refined_LocErrs']
  DATA = DATA[cols]
  # Save tracks
  pred_file = file_base + '.pred.csv'
  DATA.to_csv(pred_file, index=False)
  logging.info(f'Saved predictions: {pred_file}')


def parse_args():
  parser = argparse.ArgumentParser(
    description='Program to run ExTrack on CSV track data.')
  parser.add_argument('file', nargs='+', metavar='FILE',
    help='track data file (distances in micrometers)')

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
    default=[1, 500],
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
    default=['X', 'Y', 'Time', 'Identifier'],
    help='columns headings for X,Y,T,ID (default: %(default)s)')
  group.add_argument('--col-precision', dest='col_precision',
    # Configure for GDSC SMLM csv column heading
    default='Precision',
    help='columns heading for precision (default: %(default)s)')

  group = parser.add_argument_group('Fit parameters')
  group.add_argument('--d-max', dest='d_max', type=float,
    default=1,
    help='maximum diffusion coefficient (default: %(default)s)')
  group.add_argument('--loc-bounds', dest='loc_bounds', nargs=2, type=float,
    default=[0.01, 0.05],
    help='minimal and maximal values allowed for localization error'\
      ' parameters (default: %(default)s)')
  group.add_argument('--f-bounds', dest='f_bounds', nargs=2, type=float,
    default=[0.001, 0.99],
    help='minimum and maximum values for initial fractions'\
      ' (default: %(default)s)')
  group.add_argument('--estimated-d', dest='estimated_d', nargs='+',
    type=float, default=[0.0001, 0.03],
    help='estimated diffusion coefficients for each state'\
      ' (default: %(default)s). Note: len = nb-states'\
      ' otherwise will default to spaced between 0 and d-max')
  group.add_argument('--estimated-f', dest='estimated_f', nargs='+',
    type=float, default=[0.3, 0.7],
    help='estimated fractions for each state'\
      ' (default: %(default)s). Note: len = nb-states'\
      ' otherwise will default to 0.5')
  group.add_argument('--estimated-rate', dest='estimated_rate', nargs='+',
    type=float, default=[0.1],
    help='mean transitions per step'\
      ' (default: %(default)s). Single value or array containing state i to j,'\
      ' e.g. [k01, k02, k10, k12, k20, k21] for 3-state model.'\
      ' Note: len = nb-states * (nb-states - 1)'\
      ' otherwise will default to first value')

  group = parser.add_argument_group('Fitting')
  group.add_argument('--fit-lengths', dest='fit_lengths', nargs=2, type=int,
    default=[2, 20],
    help='track lengths considered (default: %(default)s)')
  group.add_argument('-n', '--repeats', dest='repeats', type=int, default=3,
    help='number of repeats (default: %(default)s)')
  group.add_argument('--threshold', dest='threshold', type=float, default=0.1,
    help='threshold for the fusion of the sequences of states'
      ' (default: %(default)s)')
  group.add_argument('--sub-steps', dest='substeps', type=int, default=1,
    help='number of considered transition steps in between consecutive'\
      ' 2 positions (default: %(default)s)')

  group = parser.add_argument_group('Prediction')
  group.add_argument('--position-refinement', '-p', dest='refine',
    action='store_true',
    help='do position refinement (default: %(default)s)')
  group.add_argument('--state-durations', '-s', dest='durations',
    action='store_true',
    help='do state duration histogram (default: %(default)s)')

  args = parser.parse_args()

  # validate
  logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO)

  if len(args.estimated_d) != args.nb_states:
    logging.warning('Incorrect length for estimated-d, using default')
    args.estimated_d = None
  if len(args.estimated_f) != args.nb_states:
    logging.warning('Incorrect length for estimated-f, using default')
    args.estimated_f = None
  if len(args.estimated_rate) != args.nb_states * (args.nb_states - 1):
    if len(args.estimated_rate) != 1:
      logging.warning('Incorrect length for estimated-rate, '\
        f'defaulting to {args.estimated_rate[0]}')
    args.estimated_rate = args.estimated_rate[0]

  return args


if __name__ == '__main__':
  args = parse_args()
  if args.debug:
    logging.info('Arguments:\n' +
      '\n'.join(f'  {k}={v}' for k, v in vars(args).items()))

  for path in args.file:
    process_tracks(path, args)
