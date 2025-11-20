#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import logging
import argparse
from typing import Iterable, Tuple
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
import torch.multiprocessing as mp
import shutil
from hype import MANIFOLDS, MODELS, build_model
from hype.hypernymy_eval import main as hype_eval

try:  # Prefer the optimized Cython implementation when available
    from hype.graph_dataset import BatchedDataset  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    BatchedDataset = None

th.manual_seed(42)
np.random.seed(42)


def reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best):
    chkpnt = th.load(pth, map_location='cpu')
    model = build_model(opt, chkpnt['embeddings'].size(0))
    model.load_state_dict(chkpnt['model'])

    meanrank, maprank = eval_reconstruction(adj, model)
    sqnorms = model.manifold.norm(model.lt)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'sqnorm_min': sqnorms.min().item(),
        'sqnorm_avg': sqnorms.mean().item(),
        'sqnorm_max': sqnorms.max().item(),
        'mean_rank': meanrank,
        'map_rank': maprank,
        'best': bool(best is None or loss < best['loss']),
    }


def hypernymy_eval(epoch, elapsed, loss, pth, best):
    _, summary = hype_eval(pth, cpu=True)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'best': bool(
            best is None or summary['eval_hypernymy_avg'] > best['eval_hypernymy_avg'])
        ,
        **summary
    }


def async_eval(adj, q, logQ, opt):
    best = None
    while True:
        temp = q.get()
        if temp is None:
            return

        if not q.empty():
            continue

        epoch, elapsed, loss, pth = temp
        if opt.eval == 'reconstruction':
            lmsg = reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best)
        elif opt.eval == 'hypernymy':
            lmsg = hypernymy_eval(epoch, elapsed, loss, pth, best)
        else:
            raise ValueError(f'Unrecognized evaluation: {opt.eval}')
        best = lmsg if lmsg['best'] else best
        logQ.put((lmsg, pth))


def _python_batched_dataset(
    idx: np.ndarray,
    objects: Iterable[str],
    weights: np.ndarray,
    nnegs: int,
    batch_size: int,
    burnin: bool,
    sample_dampening: float,
):
    """Minimal pure-Python fallback when the Cython dataset is unavailable.

    This keeps the same iteration contract expected by ``train.train`` but uses
    NumPy sampling instead of the multi-threaded Cython implementation. The
    sampler follows the same dampened-frequency distribution to pick negatives.
    """

    counts = np.zeros(len(objects), dtype=np.float64)
    # Estimate popularity for the negative-sampling distribution from the tail
    # indices and associated weights
    np.add.at(counts, idx[:, 1], weights)
    probs = counts ** sample_dampening
    probs = probs / probs.sum()

    num_rows = idx.shape[0]
    order = np.arange(num_rows)

    def iterator():
        np.random.shuffle(order)
        cursor = 0
        while cursor < num_rows:
            batch_idx = order[cursor : cursor + batch_size]
            cursor += batch_size
            positives = idx[batch_idx]
            batch = np.empty((positives.shape[0], nnegs + 2), dtype=np.int64)
            batch[:, 0:2] = positives
            # Draw negatives; ensure they are not equal to the tail index to
            # avoid trivial positives. Resample any collisions.
            negs = np.random.choice(len(objects), size=(positives.shape[0], nnegs), p=probs)
            collisions = negs == positives[:, 1:2]
            while collisions.any():
                negs[collisions] = np.random.choice(len(objects), size=collisions.sum(), p=probs)
                collisions = negs == positives[:, 1:2]
            batch[:, 2:] = negs
            yield th.LongTensor(batch), th.zeros(batch.shape[0]).long()

    return iterator()


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', default='/tmp/hype_embeddings.pth',
                        help='Where to store the model checkpoint')
    parser.add_argument('-dset', type=str, required=True,
                        help='Dataset identifier')
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='lorentz',
                        choices=MANIFOLDS.keys())
    parser.add_argument('-model', type=str, default='distance',
                        choices=MODELS.keys(), help='Energy function model')
    parser.add_argument('-lr', type=float, default=1000,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('-margin', type=float, default=0.1, help='Hinge margin')
    parser.add_argument('-eval', choices=['reconstruction', 'hypernymy'],
                        default='reconstruction', help='Which type of eval to perform')
    opt = parser.parse_args()

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    if opt.gpu >= 0 and opt.train_threads > 1:
        opt.gpu = -1
        log.warning(f'Specified hogwild training with GPU, defaulting to CPU...')

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')

    if 'csv' in opt.dset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        if BatchedDataset is not None:
            data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
                opt.ndproc, opt.burnin > 0, opt.dampening)
        else:
            log.warning('Falling back to the pure-Python edge-list loader because the '
                        'Cython extension is unavailable. Install the package with '
                        '`python setup.py build_ext --inplace` for faster training.')
            data = _python_batched_dataset(idx, objects, weights, opt.negs,
                opt.batchsize, opt.burnin > 0, opt.dampening)
    else:
        log.info('Using adjacency matrix dataloader')
        dset = load_adjacency_matrix(opt.dset, 'hdf5')
        log.info('Setting up dataset...')
        data = AdjacencyDataset(dset, opt.negs, opt.batchsize, opt.ndproc,
            opt.burnin > 0, sample_dampening=opt.dampening)
        objects = dset['objects']

    model = build_model(opt, len(objects))

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(), lr=opt.lr)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf' : vars(opt), 'objects' : objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    controlQ, logQ = mp.Queue(), mp.Queue()
    control_thread = mp.Process(target=async_eval, args=(adj, controlQ, logQ, opt))
    control_thread.start()

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        model.manifold.normalize(lt)

        checkpoint.path = f'{opt.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'model_type': opt.model,
        })

        controlQ.put((epoch, elapsed, loss, checkpoint.path))

        while not logQ.empty():
            lmsg, pth = logQ.get()
            shutil.move(pth, opt.checkpoint)
            if lmsg['best']:
                shutil.copy(opt.checkpoint, opt.checkpoint + '.best')
            log.info(f'json_stats: {json.dumps(lmsg)}')

    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)
    if opt.train_threads > 1:
        threads = []
        model = model.share_memory()
        args = (device, model, data, optimizer, opt, log)
        kwargs = {'ctrl': control, 'progress' : not opt.quiet}
        for i in range(opt.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=train.train, args=args, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:
        train.train(device, model, data, optimizer, opt, log, ctrl=control,
            progress=not opt.quiet)
    controlQ.put(None)
    control_thread.join()
    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')


if __name__ == '__main__':
    main()
