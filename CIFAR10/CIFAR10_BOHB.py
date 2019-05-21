# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:55:55 2019

@author: lawle
"""

import os
import pickle
import argparse
from CIFAR10_BOHB_worker import KerasWorker as worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
etha = 3
max_budget = 54
n_iterations = 10


parser = argparse.ArgumentParser(description='CIFAR10_BOHB_simple_hadd')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=max_budget)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=n_iterations)
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')

#parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=n_workers)
#parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
#parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
#parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
#parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='keras')

args=parser.parse_args()

f = open('BOHB_CIFAR10_simple_hadd_%siters_%sepochs.txt' %(args.n_iterations, args.max_budget), 'w')

# Start a nameserver:
NS = hpns.NameServer(run_id='CIFAR10_BOHB', host='127.0.0.1', port=None)
NS.start()
# NS = hpns.NameServer(run_id='CIFAR10_BOHB', host='127.0.0.1', port=0, working_directory=args.shared_directory)
# ns_host, ns_port = NS.start()


w = worker(sleep_interval = 0.5, nameserver='127.0.0.1', run_id='CIFAR10_BOHB')
# w = worker(run_id='CIFAR10_BOHB', host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)


# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
              eta = etha,
			  result_logger=result_logger,
              run_id = 'CIFAR10_BOHB',
			  min_budget=args.min_budget, max_budget=args.max_budget,
              num_samples = 10,
              min_points_in_model = 5
		   )
res = bohb.run(n_iterations=args.n_iterations)
#res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)
    
total_epochs = 0
'''
for i in range(args.n_workers):
    total_epochs = total_epochs + workers[i].total_epochs
'''
total_epochs = w.total_epochs
    
print("\nTotal Epochs : %s\n" % total_epochs)
f.write("Total Epochs : %s\n" % total_epochs)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()

all_runs = res.get_all_runs()
all_runs.sort(key=lambda r: r.time_stamps['finished'])
best_runs = list(filter(lambda r: r.budget==max_budget, all_runs))
best_runs.sort(key=lambda r: r.loss)

f.write('\nBest found configuration' + str(id2config[best_runs[0]['config_id']]['config']))
f.write('\n\nBest found accuracy: %s' % (-best_runs[0]['loss']))
f.write('\n\nA total of %i unique configurations where sampled.\n' % len(id2config.keys()))
f.write('\nA total of %i runs where executed.\n' % len(res.get_all_runs()))
f.write('\nThe run took %.1f seconds to complete.\n'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

f.close()