# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:55:55 2019

@author: lawle
"""

import os
import pickle
import argparse
from fMNIST_BOHB_worker import SVMWorker as worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
etha = 3
max_budget = 27
n_iterations = 8


parser = argparse.ArgumentParser(description='fMNIST_BOHB_parser')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=max_budget)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=n_iterations)
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')

args=parser.parse_args()

f = open('BOHB_fMNIST_%siters_%sbudgets.txt' %(args.n_iterations, args.max_budget), 'w')

# Start a nameserver:
NS = hpns.NameServer(run_id='fMNIST_BOHB', host='127.0.0.1', port=None)
NS.start()

w = worker(sleep_interval = 0.5, nameserver='127.0.0.1', run_id='fMNIST_BOHB')
w.run(background=True)

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)


# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
              eta = etha,
			  result_logger=result_logger,
              run_id = 'fMNIST_BOHB',
			  min_budget=args.min_budget, max_budget=args.max_budget,
              num_samples = 10,
              min_points_in_model = 5
		   )
res = bohb.run(n_iterations=args.n_iterations)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)
    
total_budgets = 0
total_budgets = w.total_budgets
    
print("\nTotal Epochs : %s\n" % total_budgets)
f.write("Total Epochs : %s\n" % total_budgets)

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