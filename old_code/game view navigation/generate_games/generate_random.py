import itertools
import os
import json
import numpy as np
import random as rnd
import sys
import time

def get_profiles(num_players, num_strategies):
	temp = itertools.repeat(range(num_players+1), num_strategies)
	all_agent_combos = itertools.product(*temp)
	all_profiles = filter(lambda a: list(itertools.accumulate(a))[-1] == num_players, all_agent_combos)

	return list(all_profiles)

def init_game(gid, num_players, num_strats, min_val, max_val):
	payoff_data = {}
	game_name = 'random_game_{0}'.format(gid)
	payoff_data['id'] = hash(game_name)
	payoff_data['name'] = game_name
	payoff_data['simulator_fullname'] = game_name
	payoff_data['configuration'] = [['N', num_players], ['min_val', min_val], ['max_val', max_val]]

	payoff_data['roles'] = []
	tmp_role = {}
	tmp_role['name'] = 'all'
	tmp_role['strategies'] = [str(x) for x in range(0, num_strats)]
	tmp_role['count'] = num_players
	payoff_data['roles'].append(tmp_role)

	payoff_data['profiles'] = []

	return payoff_data

def gen_payoff(payoffs, profile, min_val, max_val):
	prof_name = ''
	for p in profile:
		prof_name += str(p)

	new_prof = {}
	new_prof['id'] = prof_name
	new_prof['observations_count'] = 1
	new_prof['symmetry_groups'] = []

	for p in range(0, len(profile)):
		pay = 0
		if profile[p] > 0:
			pay = rnd.uniform(min_val, max_val)

		new_prof['symmetry_groups'].append({
			'id': prof_name + str(p),
			'role': 'all',
			'strategy': str(p),
			'count': int(profile[p]),
			'payoff': pay,
			'payoff_sd': 0
			})

	payoffs['profiles'].append(new_prof)

	return payoffs

def write_payoffs(path, gid, payoffs):
	file = os.path.join(path, 'random_game_{0}.json'.format(gid))
	f = open(file, 'w')
	json.dump(payoffs, f)
	f.close()

if __name__ == '__main__':
	
	game_path = '7_strat_random'
	strats = 7
	
	players = 25
	min_val = 0
	max_val = 1

	# what game ids to generate; max_id is inclusive
	min_id = int(sys.argv[1])
	max_id = int(sys.argv[2])

	pay_data = {}

	start = time.time()

	for id in range(min_id, max_id+1):
		print('Creating game:', id)
		now2 = time.time()

		pay_data = init_game(id, players, strats, min_val, max_val)
		profiles = get_profiles(players, strats)
		total_profs = len(profiles)
		curr_prof = 0

		for prof in profiles:
			pay_data = gen_payoff(pay_data, prof, min_val, max_val)
			curr_prof += 1
			if curr_prof%100 == 0:
				print('Wrote: {0}/{1} profiles'.format(curr_prof, total_profs))

		write_payoffs(game_path, id, pay_data)
		print('Successfully wrote game in {0}.'.format(time.time()-now2))
		pay_data = {}

	end = time.time()
	print('Creating {0} games took {1}'.format(max_id-min_id, end-start))


