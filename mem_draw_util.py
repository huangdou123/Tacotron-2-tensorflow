""" Get Memory Usage in Tensorflow (r1.15.2)

Usage:
  mem_draw_util.py [-m MODEL] [-b BATCH_SIZE] [-i ITER] [-t TRAIN]
  mem_draw_util.py -h | --help | --version

Options:
 -h --help                              show this help message and exit
 --version                              show version and exit
 -m MODEL --model=MODEL                 the model name
 -b BATCH_SIZE --batch_size=BATCH_SIZE  the batch size
 -i ITER --iter=ITER                    the iteration to draw
 -t --train <true_or_false>             train or inference[default: True]
"""

from __future__ import print_function

from docopt import docopt
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import graph

decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)


def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)


# Global Configurations

_ignore_devices = ['cpu', 'kernel', 'memcpy']

# for matplotlib color
black = '#000000'
colors = ['tomato', 'darkcyan']

# overwrite option
# as tf.estimator.train() wraps all iterations' sess.run()
# the run_metadata is either enabled or disabled for all iterations
# for bert this overwrite is enabled
# deprecated
_overwrite = False
_except_nets = ['bert', 'mobilenet_v2', 'mobilenet_v3']


def _bad_devices(device_name):
  for i in _ignore_devices:
    if i in device_name:
      return True

  return False


def _simplify_device_name(device):
  """/job:localhost/replica:0/task:0/device:CPU:0 -> /cpu:0"""

  prefixs = [
      '/job:localhost/replica:0/task:0/device:',
      '/device:',
      # new in tf r1.15.2
      '/gpu:0 (Tesla P100-PCIE-16GB)/context#0',  # hack for P100 GPU
      '/host:'
  ]

  parse = False
  for prefix in prefixs:
    if device.startswith(prefix):
      device = device[len(prefix):]
      parse = True
      break

  if not parse:
    eprint('Can not parse the device name: {}'.format(device))
    exit(1)

  if '/' in device:
    device = '_'.join(device.split('/'))

  if ':' in device:
    device = '_'.join(device.split(':'))

  return device.lower()


class Config(object):
  __slots__ = (
      'out_dir',
      'figure_dir',
      'mem_f',
      'net_name',
      'bs',
      'step_id',
      'run_prefix',
      'uname',
      'alloc_f',
      'temp_mem_f',
      'pers_mem_f',
      'new_net',
  )

  def __init__(self, **kwargs):
    super(Config, self).__init__()
    self.out_dir = './graph'
    self.figure_dir = './figures'
    self.mem_f = './graph/mem.json'

    self.net_name = None
    self.bs = -1
    self.step_id = -1
    self.run_prefix = ''
    self.uname = None  # a unique name '{run_prefix}_{net_name}_{bs}'

    # dump file name
    # FileName: <train_or_eval>_netname_batchsize_stepid_alloc.log: (allocation_time, allocation_bytes)
    self.alloc_f = ''

    # these two files only write once
    # FileName: <train_or_eval>_netname_batchsize_tmpmem.log: (memory_size)
    self.temp_mem_f = ''
    # FileName: <train_or_eval>_netname_batchsize_persmem.log: (memory_size)
    self.pers_mem_f = ''

    # if running a new network: treat multiple iterations of the same networks as the same net
    self.new_net = True

    for f, v in kwargs.items():
      setattr(self, f, v)

    if self.net_name:
      self.uname = '{}_{}_{}'.format(self.run_prefix, self.net_name, self.bs)

      self.out_dir = self.out_dir + '/{}'.format(self.net_name)
      self.figure_dir = self.figure_dir + '/{}'.format(self.net_name)

      if not os.path.exists(self.out_dir):
        os.mkdir(self.out_dir)
      if not os.path.exists(self.figure_dir):
        os.mkdir(self.figure_dir)

      self.alloc_f = '{}/{}_{}_{}_{}_alloc.log'.format(self.out_dir,
                                                       self.run_prefix,
                                                       self.net_name, self.bs,
                                                       self.step_id)
      self.temp_mem_f = '{}/{}_{}_{}_tmpmem.log'.format(
          self.out_dir, self.run_prefix, self.net_name, self.bs)
      self.pers_mem_f = '{}/{}_{}_{}_persmem.log'.format(
          self.out_dir, self.run_prefix, self.net_name, self.bs)

      if os.path.exists(self.temp_mem_f) and os.path.exists(self.pers_mem_f):
        self.new_net = False
    else:
      eprint('name of networks should not be None')

  def issame(self, netname):
    return netname == self.uname


class MemInfo(object):
  __slots__ = (
      'netname',
      'peak_mem',
      'temp_mem',
      'pers_mem',
      'mem_alloc_num',
      'temp_alloc_num',
      'pers_alloc_num',
  )

  def __init__(self, **kwargs):
    super(MemInfo, self).__init__()
    self.netname = None
    self.peak_mem = 0
    self.temp_mem = 0
    self.pers_mem = 0
    self.mem_alloc_num = 0
    self.temp_alloc_num = 0
    self.pers_alloc_num = 0

    for f, v in kwargs.items():
      setattr(self, f, v)

  def __repr__(self):
    content = ', '.join([
        '\"{}\" : \"{}\"'.format(f, getattr(self, f))
        for f in MemInfo.__slots__
    ])
    return '{' + content + '}'


def get_alloc_infos(net,
                    batch_size,
                    step_id,
                    is_train,
                    run_metadata,
                    draw=False):
  assert run_metadata != None
  assert hasattr(run_metadata, 'step_stats')
  assert hasattr(run_metadata.step_stats, 'dev_stats')

  run_prefix = 'train' if is_train else 'eval'
  cfg = Config(net_name=net,
               bs=batch_size,
               step_id=step_id,
               run_prefix=run_prefix)

  if net in _except_nets:
    _overwrite = True  # deprecated
    # print('Overwrite each step metadata!')

  dev_stats = run_metadata.step_stats.dev_stats
  g = graph.Graph(cfg)
  for dev_stat in dev_stats:
    # print('original device name: {}'.format(dev_stat.device))
    device_name = _simplify_device_name(dev_stat.device)
    # print('after simplfy: {}'.format(device_name))
    # pass the graph in cpu
    # devices name with 'stream' record the right execution time,
    # but the mem allocation is not in them
    if _bad_devices(device_name):
      continue

    if 'stream_all' in device_name:
      # process gpu_0_stream_all: accurate gpu time of node execution
      g.InitNodeTime(dev_stat.node_stats)
    elif device_name.startswith('_stream#'):
      pass
    else:
      # process gpu_0: contain nodestat.output (cpu time)
      g.InitNodeCPUTime(dev_stat.node_stats)
      # _dump_one_step(nodestats=dev_stat.node_stats, cfg=cfg, draw=False)
      # _debug_mem_per_node(nodestats=dev_stat.node_stats, cfg=cfg)
      # these two files are only necessary to be initialized once
      g.InitNodeOutputs(dev_stat.node_stats)
      g.InitNodeAllocations(dev_stat.node_stats)

  # g.diff(cfg)
  # g.CheckShadowNodes()
  g.DumpToFile()
  # res = g.InitNodeInputs()
  # if res:
  #   g.InitTempAndPersTensors()
  #   g.InitSharedTensors()
  #   g.GetMemUsage()
  # g.DumpToFile()


####################### debug ########################
######################################################

# def _dump_stream_all(nodestats, filename):
#     with open(filename, 'w') as fout:
#         for node_stat in nodestats:
#             # nodename format:
#             # tower_0/v/cg/conv0/conv2d/Conv2D:Conv2D#id=26,device=/job:localhost/replica:0/task:0/device:GPU:0,async=false#@@void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)	1599121609884215	9
#             node_name = node_stat.node_name.split(':')[0]
#             all_start_micros = node_stat.all_start_micros
#             all_end_rel_micros = node_stat.all_end_rel_micros
#             fout.write('{}\t{}\t{}\n'.format(
#                 node_name, all_start_micros, all_end_rel_micros))

# for debug
# def _dump_gpu(nodestats, filename):
#     with open(filename, 'w') as fout:
#         for node_stat in nodestats:
#             fout.write('{}\n'.format(node_stat.node_name))
#             for i in node_stat.output:
#                 slot = i.slot

#                 ad = i.tensor_description.allocation_description
#                 alloc_bytes = ad.allocated_bytes
#                 allocator_name = ad.allocator_name
#                 fout.write('{}\t{}\t{}\n'.format(
#                     slot, alloc_bytes, allocator_name))

# def _test(nodestats):
#     total_nodes_num = len(nodestats)
#     no_ref_t = 0
#     tmp_f = '/tmp/tmp_ref.log'
#     # print('write the ref file: {}'.format(tmp_f))
#     # with open(tmp_f, 'w') as fout:
#     for node_stat in nodestats:
#         node_name = node_stat.node_name
#         if ':' in node_name:
#             # print >> sys.stderr, 'node name: {}'.format(node_name)
#             eprint('node name: {}'.format(node_name))
#             exit(1)

#         if len(node_stat.referenced_tensor) == 0:
#             no_ref_t += 1
#             # eprint('node name without output: {}'.format(node_name))

#     if no_ref_t == total_nodes_num:
#         eprint('error device')
#         exit(1)
# fout.write('SrcNode: {}\n'.format(node_name))
# for i in node_stat.referenced_tensor:
#   ad = i.tensor_description.allocation_description

#   alloc_bytes = ad.allocated_bytes
#   alloc_name = ad.allocator_name
#   single_ref = ad.has_single_reference  # to see if this tensor will be released when this node finishes
#   fout.write('{}\t{}'.format(alloc_bytes, 'true' if single_ref else 'false'))

# def _debug_mem_per_node(nodestats, cfg):
#     dump_f = '{}/{}{}'.format(cfg.out_dir, cfg.uname, '_mem_per_node.log')

#     total_mem = 0.0
#     with open(dump_f, 'w') as fout:
#         for node_stat in nodestats:
#             fout.write('Src node: {}\n'.format(node_stat.node_name))
#             for mem in node_stat.memory:
#                 if mem.allocator_name.lower() != 'gpu_0_bfc':
#                     continue
#                 fout.write('\t{}\t{}\t{}\n'.format(float(mem.total_bytes)/(1 << 20),
#                                                    float(mem.peak_bytes) /
#                                                    (1 << 20),
#                                                    float(mem.live_bytes)/(1 << 20)))
#                 i = 0
#                 for alloc_rd in mem.allocation_records:
#                     alloc_bytes = float(alloc_rd.alloc_bytes) / (1 << 20)
#                     if alloc_bytes > 0:
#                         fout.write('\t\t{}\t{}\n'.format(i, alloc_bytes))
#                         i += 1
#                         total_mem += alloc_bytes

# print('debug mem per node, total memory: {}'.format(total_mem))

####################### debug ########################
######################################################


def _dump_one_step(nodestats, cfg, draw=False):
  allocs_info = []
  temp_mem = []
  pers_mem = []
  allocs_mem = []

  for node_stat in nodestats:
    node_name = node_stat.node_name
    if ':' in node_name:
      node_name = node_name.split(':')[0]

    # only to record these infos when meeting a new network
    # as these don't change across iterations
    if cfg.new_net:
      mem_stat = node_stat.memory_stats
      tmp_mem = mem_stat.temp_memory_size
      persistent_mem = mem_stat.persistent_memory_size

      if tmp_mem != 0:
        temp_mem.append(float(tmp_mem) / (1 << 20))
      if persistent_mem:
        pers_mem.append(float(persistent_mem) / (1 << 20))

    # if debug:
    #     print("{}: temp mem: {}, persistent mem: {}".format(node_name, temp_mem, persistent_mem))

    alloc_memused = node_stat.memory
    # print("node_stat.memory size: {}".format(len(alloc_memused)))
    for alloc_mem in alloc_memused:
      alloc_name = alloc_mem.allocator_name
      if alloc_name.lower() != 'gpu_0_bfc':
        continue

      for alloc_rd in alloc_mem.allocation_records:
        # NOTE: this allocation time is CPU time, not GPU time
        alloc_micros = alloc_rd.alloc_micros
        alloc_bytes = alloc_rd.alloc_bytes
        allocs_info.append((alloc_micros, alloc_bytes))

  allocs_info.sort(key=lambda x: x[0])
  mem_alloc = []
  start = allocs_info[0][0]
  curr_mem = 0.0

  for _, data in enumerate(allocs_info):
    t = data[0] - start
    allocation_mem = float(data[1]) / (1 << 20)  # MB
    if allocation_mem > 0:
      allocs_mem.append(allocation_mem)
    curr_mem = curr_mem + allocation_mem / (1 << 10)
    mem_alloc.append((t, curr_mem))

  x = [d[0] for d in mem_alloc]
  y = [d[1] for d in mem_alloc]

  print('[GPU:0] Total allocated memory: {}'.format(sum(allocs_mem)))
  with open(cfg.alloc_f, 'w') as fout:
    for t, b in allocs_info:
      fout.write('{}\t{}\n'.format(t, b))

  title = cfg.uname
  if cfg.new_net:
    # only record the temporary, persistent memory and meminfo for new networks
    with open(cfg.temp_mem_f, 'w') as fout:
      for m in temp_mem:
        fout.write('{}\n'.format(m))

    with open(cfg.pers_mem_f, 'w') as fout:
      for m in pers_mem:
        fout.write('{}\n'.format(m))

    meminfo = MemInfo(netname=title,
                      peak_mem=max(y),
                      temp_mem=sum(temp_mem),
                      pers_mem=sum(pers_mem),
                      mem_alloc_num=len(allocs_mem),
                      temp_alloc_num=len(temp_mem),
                      pers_alloc_num=len(pers_mem))

    maybewrite(cfg.mem_f, meminfo)

    _plot_alloc_cdf(title, allocs_mem, cfg.figure_dir)
    _plot_mem_cdf(title, temp_mem, pers_mem, cfg.figure_dir)

  if draw:
    # print('plot alloc for step: {}'.format(cfg.step_id))
    _plot_alloc(title + '_{}'.format(cfg.step_id),
                x=x,
                y=y,
                fig_dir=cfg.figure_dir)

# deprecated no use
def _get_alloc(device_name, nodestats, dumpfile=False):
  allocs_info = []
  temp_mem = []
  pers_mem = []
  allocs_mem = []
  i = 0
  for node_stat in nodestats:
    node_name = node_stat.node_name
    if ':' in node_name:
      node_name = node_name.split(':')[0]

    mem_stat = node_stat.memory_stats
    tmp_mem = mem_stat.temp_memory_size
    persistent_mem = mem_stat.persistent_memory_size

    if tmp_mem != 0:
      temp_mem.append(float(tmp_mem) / (1 << 20))
    if persistent_mem:
      pers_mem.append(float(persistent_mem) / (1 << 20))

    # if debug:
    #     print("{}: temp mem: {}, persistent mem: {}".format(node_name, temp_mem, persistent_mem))

    alloc_memused = node_stat.memory
    # print("node_stat.memory size: {}".format(len(alloc_memused)))
    for alloc_mem in alloc_memused:
      alloc_name = alloc_mem.allocator_name
      if alloc_name.lower() != 'gpu_0_bfc':
        continue

      for alloc_rd in alloc_mem.allocation_records:
        alloc_micros = alloc_rd.alloc_micros
        alloc_bytes = alloc_rd.alloc_bytes
        allocs_info.append((alloc_micros, alloc_bytes))

  allocs_info.sort(key=lambda x: x[0])
  mem_alloc = []
  start = allocs_info[0][0]
  curr_mem = 0.0
  for _, data in enumerate(allocs_info):
    t = data[0] - start
    allocation_mem = float(data[1]) / (1 << 20)  # MB
    if allocation_mem > 0:
      allocs_mem.append(allocation_mem)
    curr_mem = curr_mem + allocation_mem / (1 << 10)  # GB
    mem_alloc.append((t, curr_mem))
    # t -= start
    # assert(t >= 0)
    # m = curr_mem + float(m) / (1<<30) # to gigabytes

  title = '{}_{}'.format(net_name, bs)
  x = [d[0] for d in mem_alloc]
  y = [d[1] for d in mem_alloc]
  alloc_num = len(allocs_mem)

  meminfo = MemInfo(netname=title,
                    peak_mem=max(y),
                    temp_mem=sum(temp_mem),
                    pers_mem=sum(pers_mem),
                    mem_alloc_num=alloc_num,
                    temp_alloc_num=len(temp_mem),
                    pers_alloc_num=len(pers_mem))
  maybewrite(meminfo)
  _plot_alloc(title=title, x=x, y=y)
  _plot_alloc_cdf(title, allocs_mem)
  _plot_cdf(title, temp_mem, pers_mem)
  # labels = ['temporary memoy', 'persistent memory']
  # _plot_cdf(title, labels, p1=temp_mem, p2=pers_mem)

  if dumpfile:
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    # alloc_f = '{}/{}_{}_{}_alloc.log'.format(out_dir, net_name, bs, stepid)
    # temp_mem_f = '{}/{}_{}_{}_tmp_mem.log'.format(out_dir, net_name, bs, stepid)
    # pers_mem_f = '{}/{}_{}_{}_pers_mem.log'.format(out_dir, net_name, bs, stepid)
    alloc_f = '{}/{}_{}_alloc.log'.format(out_dir, net_name, bs)
    temp_mem_f = '{}/{}_{}_tmp_mem.log'.format(out_dir, net_name, bs)
    pers_mem_f = '{}/{}_{}_pers_mem.log'.format(out_dir, net_name, bs)
    if _overwrite or not os.path.exists(alloc_f):
      with open(alloc_f, 'w') as fout:
        for t, b in mem_alloc:
          fout.write('{}\t{}\n'.format(t, b))

    if _overwrite or not os.path.exists(temp_mem_f):
      with open(temp_mem_f, 'w') as fout:
        for m in temp_mem:
          fout.write('{}\n'.format(m))

    if _overwrite or not os.path.exists(pers_mem_f):
      with open(pers_mem_f, 'w') as fout:
        for m in pers_mem:
          fout.write('{}\n'.format(m))


def _plot_alloc(title, x, y, fig_dir, gpu_time=True):
  if gpu_time:
    alloc_f = '{}/{}_alloc_gpu.pdf'.format(fig_dir, title)
  else:
    alloc_f = '{}/{}_alloc.pdf'.format(fig_dir, title)
  # if not _overwrite and os.path.exists(alloc_f):
  #     return

  plt.scatter(x, y, s=5, c=black)
  plt.title(title)
  plt.ylabel('GPU Memory Usage (GB)')
  plt.xlabel('Time (s)')

  plt.savefig(alloc_f, format='pdf')
  plt.clf()


def _plot_alloc_cdf(title, data, fig_dir):
  alloc_cdf_f = '{}/{}_alloc_CDF.pdf'.format(fig_dir, title)
  if not _overwrite and os.path.exists(alloc_cdf_f):
    return

  data.sort()
  count = len(data)
  data_y = list(map(lambda x: float(x) / count, range(len(data))))
  plt.plot(data, data_y, color=black, linewidth=2.0, label='allocation memory')
  plt.xlabel('allocation memory')
  plt.title(title)
  plt.legend(loc='best')
  plt.savefig(alloc_cdf_f, format='pdf')
  plt.clf()


def _plot_mem_cdf(title, a, b, fig_dir):
  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(fig_dir, title)
  if not _overwrite and os.path.exists(mem_cdf_f):
    return

  fig, ax1 = plt.subplots()

  a.sort()
  b.sort()
  count = len(a)
  data_y = list(map(lambda x: float(x) / count, range(len(a))))
  # plt.plot(a, data_y, color='tomato', linewidth=2.0, label='temp memory')
  lg1, = ax1.plot(a,
                  data_y,
                  color='tomato',
                  linewidth=2.0,
                  label='temp memory')
  ax1.set_xlabel('temporary memory')
  ax1.set_ylabel(title)

  ax2 = ax1.twiny()
  count = len(b)
  data_y = list(map(lambda x: float(x) / count, range(len(b))))
  # plt.plot(b, data_y, color='darkcyan', linewidth=2.0, label='persistent memory')
  lg2, = ax2.plot(b,
                  data_y,
                  color='darkcyan',
                  linewidth=2.0,
                  label='persistent memory')
  ax2.set_xlabel('persistent memory')
  # plt.title(title)
  # plt.text
  # plt.xlabel('Memory size (MB)')
  # fig.suptitle(title, x=0, y=0.5, ha='left', va='center')

  # fig.legend(loc='center')
  plt.legend([lg1, lg2], ['temp memory', 'persistent memory'], loc='best')
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()
  # plt.legend(loc='best')
  # plt.savefig(mem_cdf_f, format='pdf')
  # plt.clf()

def _plot_alloc_sstep(**kwargs):
  cfg = Config(net_name=kwargs['net_name'],
               bs=kwargs['bs'],
               step_id=kwargs['step'],
               run_prefix=kwargs['run_prefix'])
  gpu_time = kwargs['gpu_time']

  allocs_info = []
  x = []
  y = []
  filename = '{}/{}_{}_allocs_gpu_time.log'.format(cfg.out_dir, cfg.uname, cfg.step_id)

  with open(filename) as fin:
    for line in fin:
      tmp = line.split('\t')
      allocs_info.append((float(tmp[0])/1e6, float(tmp[1])))

  start = allocs_info[0][0]
  curr_mem = 0.0

  for data in allocs_info:
    t = data[0] - start
    curr_mem = curr_mem + float(data[1]) / (1 << 10)  # GB
    x.append(t)
    y.append(curr_mem)

  _plot_alloc(title=cfg.uname+'_'+str(cfg.step_id), x=x, y=y, fig_dir=cfg.figure_dir, gpu_time=gpu_time)
  


def _plot_alloc_msteps(**kwargs):
  cfg = Config(net_name=kwargs['net_name'],
               bs=kwargs['bs'],
               run_prefix=kwargs['run_prefix'])
  gpu_time = kwargs['gpu_time']

  iters = [i for i in range(10, 20)]

  allocs_info = []
  x = []
  y = []
  for i in iters:
    if gpu_time:
      filename = '{}_{}_allocs_gpu_time.log'.format(cfg.uname, i)
    else:
      filename = '{}_{}_alloc.log'.format(cfg.uname, i)
    with open('{}/{}'.format(cfg.out_dir, filename)) as fin:
      for line in fin:
        tmp = line.split('\t')
        allocs_info.append((float(tmp[0])/1e6, float(tmp[1])))

  # allocs_info.sort(key=lambda x: x[0])
  start = allocs_info[0][0]
  curr_mem = 0.0

  for data in allocs_info:
    t = data[0] - start
    curr_mem = curr_mem + float(data[1]) / (1 << 10)  # GB
    x.append(t)
    y.append(curr_mem)

  _plot_alloc(title=cfg.uname, x=x, y=y, fig_dir=cfg.figure_dir, gpu_time=gpu_time)


# deprecated no use
def _plot_cdf_error(title, labels, **kwargs):
  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(figure_dir, title)
  # if not _overwrite and os.path.exists(mem_cdf_f):
  #   return

  datas = kwargs.values()
  num = len(datas)

  ax = [0 for _ in range(num)]
  lg = [0 for _ in range(num)]
  fig, ax[0] = plt.subplots()

  for i, d in enumerate(datas):
    print(i, len(d))
    d.sort()
    count = len(d)
    if i > 0:
      # ax.append(ax[0].twiny())
      ax[i] = ax[0].twiny()

    data_y = list(map(lambda x: float(x) / count, range(len(d))))
    lg[i], = ax[i].plot(d,
                        data_y,
                        color=colors[i],
                        linewidth=2.0,
                        label=labels[i])
    ax[i].set_xlabel(labels[i])

    if i == 0:
      ax[i].set_ylabel(title)

  # can not place the legend in the 'best' location,
  plt.legend(lg, labels, loc='best')
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()


def _plot_alloc_f(title, filename):
  x = []
  y = []

  with open(filename) as fin:
    for line in fin:
      tmp = line.split('\t')
      x.append(int(tmp[0]))
      y.append(float(tmp[1]))

  plt.scatter(x, y, s=5, c=black)
  plt.title(title)
  plt.ylabel('GPU Memory Usage (GB)')

  alloc_f = '{}/{}_alloc.pdf'.format(figure_dir, title)
  plt.savefig(alloc_f, format='pdf')
  plt.clf()


def _plot_mem_f(title, temp_filename, pers_filename):
  temp_mem = []
  pers_mem = []
  with open(temp_filename) as fin:
    for line in fin:
      temp_mem.append(float(line))

  with open(pers_filename) as fin:
    for line in fin:
      pers_mem.append(float(line))

  temp_mem.sort()
  pers_mem.sort()

  fig, ax1 = plt.subplots()
  count = len(temp_mem)
  data_y = list(map(lambda x: float(x) / count, range(len(temp_mem))))
  lg1, = ax1.plot(temp_mem,
                  data_y,
                  color='tomato',
                  linewidth=2.0,
                  label='temp memory')
  ax1.set_xlabel('temporary memory')
  ax1.set_ylabel(title)
  # plt.legend(loc='best')

  ax2 = ax1.twiny()
  count = len(pers_mem)
  data_y = list(map(lambda x: float(x) / count, range(len(pers_mem))))
  lg2, = ax2.plot(pers_mem,
                  data_y,
                  color='darkcyan',
                  linewidth=2.0,
                  label='persistent memory')
  ax2.set_xlabel('persistent memory')

  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(figure_dir, title)
  # fig.legend(loc='best')
  plt.legend([lg1, lg2], ['temp memory', 'persistent memory'], loc='best')
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()


def test1(model, bs, is_train):
  cfg = Config(net_name=model,
               bs=bs,
               run_prefix='train' if is_train else 'eval')

  g = graph.Graph(cfg)
  g.InitFromFile()
  g.GetMemUsage(_plot_alloc)
  # g.InitNodeInputs(cfg)
  # todo1: identify the tensors with the same buffer
  # todo2: calculate the peak mem & mem usage plotting


def draw_allocs(model, batch_size, is_train, gpu_time=True, step=-1):
  configs = {
      'alexnet': 512,
      'vgg16': 64,
      'inception3': 64,
      'inception4': 64,
      'resnet50': 64,
      'resnet152': 64,
  }

  run_prefix = 'train' if is_train else 'eval'

  if model == None:
    for c in configs.items():
      _plot_alloc_msteps(net_name=c[0],
                         bs=c[1],                        
                         run_prefix=run_prefix,
                         gpu_time=gpu_time)

  else:
    if step != -1:
      _plot_alloc_sstep(net_name=model,
                        bs=configs[model],
                        step=step,
                        run_prefix=run_prefix,
                        gpu_time=gpu_time)
    else:
      _plot_alloc_msteps(net_name=model,
                         bs=configs[model],
                         run_prefix=run_prefix,
                         gpu_time=gpu_time)

  # for c in configs.items():
  #   title = '{}_{}'.format(c[0], c[1])
  #   alloc_f = '{}/{}_alloc.log'.format(out_dir, title)
  #   temp_mem_f = '{}/{}_tmp_mem.log'.format(out_dir, title)
  #   pers_mem_f = '{}/{}_pers_mem.log'.format(out_dir, title)
  #   _plot_alloc_f(title, alloc_f)
  #   _plot_mem_f(title, temp_mem_f, pers_mem_f)


# def plot_file(filename):
#   x = []
#   y = []
#   with open(filename) as fin:
#     for line in fin:
#       temp = line.split('\t')
#       x.append(float(temp[0]))
#       y.append(float(temp[1]))

#   title = '_'.join(filename.split('/')[-1].split('_')[:2])
#   # print(title)
#   _plot_alloc(title, x=x, y=y)
#   maybewrite(mem_f, (title, max(y)))


# write to file if pair.key doesn't exist in file
def maybewrite(filename, meminfo):
  mem_info = []
  if os.path.exists(filename):
    with open(filename) as fin:
      mem_info = json.load(fin, object_pairs_hook=collections.OrderedDict)

  new = True
  for it in mem_info:
    if meminfo.netname == it['netname']:
      new = False

  if new:
    # print(eval(repr(meminfo)))
    # mem_info.append(eval(repr(meminfo)))
    # print(decoder.decode(repr(meminfo)))
    mem_info.append(decoder.decode(repr(meminfo)))
    with open(filename, 'w') as fout:
      fout.write(json.dumps(mem_info, indent=2))


def test_plot():
  x = np.random.randn(20)
  y = np.random.randn(20)

  plt.scatter(x, y, s=5)
  plt.savefig('{}/{}.pdf'.format(figure_dir, 'test_scatter'))
  plt.clf()


if __name__ == "__main__":
  # draw_file()
  args = docopt(__doc__, version='0.1')
  model = args['--model']
  batch_size = args['--batch_size']
  if args['--iter'] != None:
    step = args['--iter']
  else:
    step = -1
  is_train = True if args['--train'] == 'True' else False


  # test1(model, batch_size, is_train)
  draw_allocs(model=model, batch_size=batch_size, 
              is_train=is_train, gpu_time=True, step=step)
