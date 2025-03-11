import itertools
import random as rnd
import os
import json
import time
from generate_game import generate_sin_game

#initialize a quiesce game file
def init_game_file(players, strats, functions, gid):
	pay_data = {}
	pay_data['id'] = gid
	pay_data['name'] = 'BAGG-FNA-game'
	pay_data['simulator_fullname'] = 'BAGG-FNA_simulator'
	pay_data['configuration'] =  [['N', players, 'num_functions', functions]]
	pay_data['roles'] = [{'name': 'all', 'count': players, 'strategies': strats}]
	pay_data['profiles'] = []

	return pay_data

#add a profile's payoff data to the quiesce game file
def add_game_file(profile, payoff, game_data, strats):
	new_data = {}
	new_data['id'] = rnd.randint(1, 100000)
	new_data['observations_count'] = 1
	new_data['symmetry_groups'] = []

	for p in range(0, len(profile)):
		tmp = {}
		tmp['id'] = p
		tmp['role'] = 'all'
		tmp['strategy'] = strats[p]
		tmp['count'] = profile[p]
		if profile[p] == 0:
			tmp['payoff'] = 0
		else:
			tmp['payoff'] = payoff[p]
		tmp['payoff_sd'] = 0
		new_data['symmetry_groups'].append(tmp)

	game_data['profiles'].append(new_data)

	return game_data

#get the payoff for a given profile and add the payoff to the data dictionary
def gen_payoff(game, profile, data, strats):
	pay = game.get_payoffs2(profile)
	data = add_game_file(profile, pay, data, strats)

	return data

#get all the profiles of the game
def get_all_profiles(players, strategies):
	strats = [i for i in range(0, len(strategies))]
	temp = itertools.repeat(range(players+1), len(strats))
	all_combos = itertools.product(*temp)
	all_profiles = filter(lambda a: list(itertools.accumulate(a))[-1] == players, all_combos)

	return list(all_profiles)

def get_game(num_games, num_players, num_strats, num_funcs):
	game = generate_sin_game(num_players, num_strats, num_funcs)
	strats = game.action_names
	profs = get_all_profiles(num_players, strats)

	return game,profs,strats

if __name__ == '__main__':

	path = 'games/'
	num_games = 100
	num_players = 49
	num_strats = 5
	num_funcs = 7

	for g in range(0, num_games):
		game,profiles,strategies = get_game(num_games, num_players, num_strats, num_funcs)
		pay_data = init_game_file(num_players, strategies, num_funcs, g)

		for profile in profiles:
			pay_data = gen_payoff(game, profile, pay_data, strategies)

		f = open(os.path.join(path, 'game_{0}.json'.format(g)), 'w')
		json.dump(pay_data, f)
		f.close()
		print('Wrote game {0}/{1}'.format(g+1, num_games))

	print('Done.')