import argparse
import os
import numpy as np
import time
import json
import itertools
from collections import Counter

import multiprocessing as mp
import asyncio

import quiesce_innerloop.innerloop as il
from gameanalysis import gamereader as gread
from gameanalysis import restrict

"""
minimum constrained regret profile search for random game experiment
"""

#read in the list of profiles from the designated directory
def get_profiles(file):
	f = open(file, 'r')
	profs = f.readlines()
	f.close()

	all_profs = []

	for p in profs:
		all_profs.append(tuple(eval(p)))

	return all_profs

def get_regret(game, profile):
	gains,dev_profs = il.mixture_deviation_gains(game, profile)
	role_gains = np.fmax.reduceat(gains, game.role_starts)
	gain = np.nanmax(role_gains)

	return gain,dev_profs

#approximate a profile as a mixed strategy equilibrium
def transform_to_meq(profs):
	num_play = np.sum(profs[0])
	new_profs = []

	for pr in profs:
		tmp_new_prof = []

		for i in range(0, len(pr)):
			tmp_new_prof.append(pr[i]/num_play)

		new_profs.append(tmp_new_prof)

	return new_profs

"""
# regular version
#proc_profs should be mixed equilibria, seen_profs should be agent counts
def run_mrcp(game, proc_profs, seen_profs, get_total=False, mixed=True):
	min_reg = 100000
	min_prof = []
	curr_prof = 0
	total_prof = len(proc_profs)

	for pr in proc_profs:
		#print('Profile:', pr)
		# if the profiles we're reading in aren't mixed (probability), then turn them into those
		if mixed == False:
			pr_mixed = transform_to_meq([pr])[0]
			#print('Mixed version:', pr_mixed)
			tmp_reg,tmp_profs = get_regret(game, pr_mixed)
		else:
			tmp_reg,tmp_profs = get_regret(game, pr)

		for t in tmp_profs:
			if tuple(t) not in seen_profs:
				seen_profs.append(tuple(t))

		if tmp_reg <= min_reg:
			#we've found a new min, old stored mins should be overridden
			if tmp_reg < min_reg:
				min_reg = tmp_reg
				min_prof = []

			#seen_profs = list(set(seen_profs))
			num_visited = len(seen_profs)
			min_prof.append(tuple([pr, tmp_reg, num_visited]))

		curr_prof += 1
		if curr_prof%100 == 0:
			print('Currently: {0}/{1}'.format(curr_prof, total_prof))

	# so it doesn't mess up existing methods
	if get_total == True:
		final_visited = len(seen_profs)
		min_prof.append(tuple(['final total visited', 0, final_visited]))

	return min_prof
"""

# set and dictionary -- faster
def run_mrcp(game, proc_profs, seen_profs, get_total=False, mixed=True):
	proc = {}
	total_profs = len(proc_profs)
	curr_profs = 0
	seen_profs = set(seen_profs)

	for pr in proc_profs:
		if mixed == False:
			pr_mixed = transform_to_meq([pr])[0]
			tmp_reg,tmp_profs = get_regret(game, pr_mixed)
		else:
			tmp_reg,tmp_profs = get_regret(game, pr)

		for t in tmp_profs:
			seen_profs.add(tuple(t))
		if tmp_reg not in proc:
			proc[tmp_reg] = []
		proc[tmp_reg].append([pr, len(seen_profs)])

		curr_profs += 1
		if curr_profs%100 == 0:
			print('Processing profiles: {0}/{1}'.format(curr_profs, total_profs))

	print('Finding min.')
	min_reg = np.min(list(proc))
	min_prof = []
	for p in proc[min_reg]:
		min_prof.append(tuple([p[0], min_reg, p[1]]))
	min_prof.append(tuple(['final visited', 0, len(seen_profs)]))

	return min_prof

#writes to cummulative data file that saves only regret and profile counts
def write_data(data_path, gid, setting, new_data, method='m1'):
	"""
	file = os.path.join(data_path, 'new_mrcp_{0}_data.json'.format(method))

	if os.path.exists(file):
		f = open(file, 'r')
		curr_data = json.load(f)
		f.close()
	else:
		curr_data = {}

	if str(setting) not in curr_data:
		curr_data[str(setting)] = {}
	curr_data[str(setting)][str(gid)] = new_data
	
	f = open(file, 'w')
	json.dump(curr_data, f)
	f.close()
	"""

	#write individual game info
	file = os.path.join(data_path, str(gid), 'mrcp_{0}_{1}.txt'.format(method, str(setting)))
	f = open(file, 'w')
	for nd in new_data:
		if nd[0] == 'final total visited':
			f.write('Total profiles at end of search: {0}\n\n'.format(nd[2]))
		else:
			f.write('Equilibrium: {0}\n'.format(nd[0]))
			f.write('Regret: {0}\n'.format(nd[1]))
			f.write('Number of profiles: {0}\n\n'.format(nd[2]))
	f.close()

# just dumps the data in a file
def write_data2(data_path, gid, setting, new_data, method='m1'):
	if not os.path.exists(os.path.join(data_path, str(gid))):
		os.mkdir(os.path.join(data_path, str(gid)))

	file = os.path.join(data_path, str(gid), 'mrcp_{0}_{1}.txt'.format(method, str(setting)))
	#file = os.path.join(data_path, str(gid), 'test_new_mrcp_{0}_{1}.txt'.format(method, str(setting)))
	f = open(file, 'w')
	f.write(str(new_data))
	f.close()

"""
different versions of mrcp for random games
"""
#use profiles from quiesce+dpr and full game regret -- final method 3
def m1(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	mixed = transform_to_meq(seen_profiles)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	#agg
	#gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	#f = open(gfile, 'r')
	f = open(game_path, 'r')
	game = gread.load(f)
	f.close()

	eq_profs = run_mrcp(game, mixed, seen_profiles)

	return eq_profs

"""
#use profiles from quiesce+dpr deemed beneficial and full game regret
def m2(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	pfile = os.path.join(prof_path, str(game_id), 'bene_profiles_{0}.txt'.format(str(setting)))
	proc_profiles = get_profiles(pfile)
	mixed = transform_to_meq(proc_profiles)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	f = open(gfile, 'r')
	game = gread.load(f)
	f.close()

	eq_profs = run_mrcp(game, mixed, seen_profiles)

	return eq_profs
"""

#get profiles part of a restricted game, when necessary turn the profiles to the restricted game format
def get_restricted_profiles(prof_path, game_id, setting, all_profiles, restrict=False):
	rfile = os.path.join(prof_path, str(game_id), 'dpr_rgame_{0}.txt'.format(str(setting)))
	f = open(rfile, 'r')
	data = f.readlines()
	f.close()
	rgame = eval(data[0])

	rest_profiles = []

	for ap in all_profiles:
		add = True
		if restrict == True:
			tmp_profile = []
		for i in range(0, len(ap)):
			if ap[i] > 0 and rgame[i] == False:
				add = False
			elif restrict == True:
				if rgame[i] == True:
					tmp_profile.append(ap[i])

		if add == True:
			if restrict == False:
				rest_profiles.append(ap)
			else:
				rest_profiles.append(tuple(tmp_profile))

	return rest_profiles,rgame

#turn restricted game equilibrium into full game and get full game regret
def proc_m3_equ(game, rest_eq, rgame, non_restrict):
	full_eq = []

	for eq,_,cnt in rest_eq:
		new_eq = []
		curr_in = 0
		for i in range(0, len(rgame)):
			if rgame[i] == True:
				new_eq.append(eq[curr_in])
				curr_in += 1
			else:
				new_eq.append(0)

		reg,_ = get_regret(game, new_eq)
		#all those in the non-restricted game were visited before mrcp search, so adds to the profile count
		full_eq.append(tuple([new_eq, reg, cnt+non_restrict]))

	return full_eq

"""
#uses profiles from quiesce+dpr in restricted game formed by equilibria and restricted game regret
def m3(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	#get profiles in restricted game
	use_profiles,rest_game = get_restricted_profiles(prof_path, game_id, setting, seen_profiles, restrict=True)
	#number of profiles visited in the quiesce dpr search, but not part of the subsequent restricted game
	cnt_non_restrict = len(seen_profiles) - len(use_profiles)
	mixed = transform_to_meq(use_profiles)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	f = open(gfile, 'r')
	game = gread.load(f)
	f.close()
	rgame = game.restrict(rest_game)

	eq_profs = run_mrcp(rgame, mixed, use_profiles)
	new_eq_profs = proc_m3_equ(game, eq_profs, rest_game, cnt_non_restrict)

	return new_eq_profs
"""

#uses profiles from quiesce+dpr in restricted game formed by equilibria and full game regret -- final method 4
def m4(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	#get profiles in restricted game
	use_profiles,_ = get_restricted_profiles(prof_path, game_id, setting, seen_profiles)
	mixed = transform_to_meq(use_profiles)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	#gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	#f = open(gfile, 'r')
	f = open(game_path, 'r')
	game = gread.load(f)
	f.close()

	eq_profs = run_mrcp(game, mixed, seen_profiles)

	return eq_profs

# helper method for three layer to return all profiles in the game formed by the restricted 
"""
def get_required_profiles(rest, all_strats, num_players):
	rgame_profiles = []
	avail_strats = []

	ravail = []
	for a in range(0, len(all_strats)):
		if rest[a]:
			ravail.append(a)
	for _ in range(0, num_players):
		avail_strats.append(ravail)

	all_profiles = itertools.product(*avail_strats)

	for a in all_profiles:
		a_cnts = Counter(a)
		temp_prof = np.zeros(len(rest))
		for c in a_cnts:
			temp_prof[c] = a_cnts[c]

		temp_prof = tuple(temp_prof)
		if temp_prof not in rgame_profiles and sum(temp_prof) == num_players:
			rgame_profiles.append(temp_prof)

	rgame_profiles = [list(rp) for rp in rgame_profiles]

	return rgame_profiles
"""
# single role version
def get_required_profiles(rest, all_strats, num_players):
	avail = []
	np = [i for i in range(0, num_players+1)]
	for _ in range(0, len(all_strats)):
		avail.append(np)
	all_profiles = itertools.product(*avail)
	
	good_profs = []
	for a in all_profiles:
		if sum(a) == num_players:
			add = True 
			for ai in range(len(a)):
				if a[ai] > 0 and rest[ai] == False:
					add = False
			if add == True and a not in good_profs:
				good_profs.append(a)

	good_profs = [list(gp) for gp in good_profs]

	return good_profs

# TODO: edit so that we grab all profiles from the different dprs to add to seen profiles first
# uses profiles from quiesce+dpr in restricted game formed by equilibria and full game regret
def three_layer(prof_path, game_path, game_id, setting):
	# random game
	nxt_setting = {3: 7, 4: 9, 5: 13}
	players = 25

	#print('Starting 3 layer.')

	setting2 = nxt_setting[setting]
	# for 5 player version
	#all_strats = [x for x in range(0, 5)]
	all_strats = [x for x in range(0, 7)]

	# get all seen profiles in the 2 dpr levels
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	pfile2 = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting2)))
	new_profiles = get_profiles(pfile2)
	seen_profiles.extend(new_profiles)
	seen_profiles = list(set(seen_profiles)) # make sure we have unique profiles

	# read in from file containing the restricted game formed by those strategies in support of equilibria
	efile = os.path.join(prof_path, str(game_id), 'dpr_rgame_{0}.txt'.format(str(setting2)))
	f = open(efile, 'r')
	data = f.readlines()
	f.close()
	equ_rest = eval(data[0])
	print('Game:', equ_rest)

	#gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	#f = open(gfile, 'r')
	f = open(game_path, 'r')
	game = gread.load(f)
	f.close()

	# profiles just in the restricted game formed by those in support of equilibria
	use_profiles = get_required_profiles(equ_rest, all_strats, players)
	eq_profs = run_mrcp(game, use_profiles, seen_profiles, get_total=True, mixed=False)

	return eq_profs

#uses profiles from quiesce+dpr that are candidate equilibria and full game regret -- final method 5
def m5(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	#get candidate equilibria
	eqfile = os.path.join(prof_path, str(game_id), 'candidate_eq_{0}.txt'.format(str(setting)))
	eq_cands = get_profiles(eqfile)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	#gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	#f = open(gfile, 'r')
	f = open(game_path, 'r')
	game = gread.load(f)
	f.close()

	equ_profs = run_mrcp(game, eq_cands, seen_profiles)

	return equ_profs

"""
#uses beneficial profiles from quiesce+dpr that are in the restricted game and full game regret
def m6(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	pfile = os.path.join(prof_path, str(game_id), 'bene_profiles_{0}.txt'.format(str(setting)))
	proc_profiles = get_profiles(pfile)
	#get profiles in restricted game
	use_profiles,_ = get_restricted_profiles(prof_path, game_id, setting, proc_profiles)
	mixed = transform_to_meq(use_profiles)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	f = open(gfile, 'r')
	game = gread.load(f)
	f.close()

	eq_profs = run_mrcp(game, mixed, seen_profiles)

	return eq_profs
"""

#uses candidate equilibria from quiesce+dpr that are in the restricted game and full game regret -- final method 6
def m7(prof_path, game_path, game_id, setting):
	pfile = os.path.join(prof_path, str(game_id), 'profiles_{0}.txt'.format(str(setting)))
	seen_profiles = get_profiles(pfile)
	#get candidate equilibria
	eqfile = os.path.join(prof_path, str(game_id), 'candidate_eq_{0}.txt'.format(str(setting)))
	eq_cands = get_profiles(eqfile)
	eq_cands,_ = get_restricted_profiles(prof_path, game_id, setting, eq_cands, restrict=False)

	#gfile = os.path.join(game_path, 'random_game_25_5_{0}.json'.format(str(game_id)))
	#gfile = os.path.join(game_path, 'game_{0}.json'.format(str(game_id)))
	#f = open(gfile, 'r')
	f = open(game_path, 'r')
	game = gread.load(f)
	f.close()

	equ_profs = run_mrcp(game, eq_cands, seen_profiles)

	return equ_profs

if __name__ == '__main__':

	# version for running multiple folders for method 7 (3 layer method)
	parser = argparse.ArgumentParser()
	parser.add_argument('mingame', type=int)
	parser.add_argument('maxgame', type=int) # inclusive
	args = parser.parse_args()

	#profpath = '7_strat_random/experiments/method7' # conservative
	profpath = '7_strat_random/experiments/method8' # aggressive
	write_path = '7_strat_random/experiments/mrcp_methods'
	dpr = 3

	for game in range(args.mingame, args.maxgame+1):
		gamepath = os.path.join('7_strat_random', 'random_game_{0}.json'.format(game))
		print('Running Method 7 on game', game, '.')

		eq_profs = three_layer(profpath, gamepath, game, dpr)
		#write_data2(write_path, game, dpr, eq_profs, method='multi_m7') # conservative
		write_data2(write_path, game, dpr, eq_profs, method='multi_m8') # aggressive
		print('Wrote data.')

	"""
	parser = argparse.ArgumentParser()
	#parser.add_argument('gamepath', type=str, help='Path to the directory containing the JSON game files.')
	#parser.add_argument('profpath', type=str, help='Path to the directory containing the game folders containing the profile list files.')
	parser.add_argument('game', type=int, help='Number of the game we are processing now.')
	parser.add_argument('dpr', type=int, help='Setting of DPR used when running quiesce.')
	#reductions = parser.add_mutually_exclusive_group()
	#reductions.add_argument('--rest', type=str)
	args = parser.parse_args()

	# hard coded for 7-strategy random matrix game
	gamepath = os.path.join('7_strat_random', 'random_game_{0}.json'.format(args.game))
	write_path = '7_strat_random/experiments/mrcp_methods'

	# for method 3-6
	#profpath = '7_strat_random/experiments/method2'
	# for method 7/8
	profpath = '7_strat_random/experiments/method7' # conservative
	#profpath = '7_strat_random/experiments/method8' # aggressive

	# USE
	print('Method 7:') # new multilevel
	eq_profs = three_layer(profpath, gamepath, args.game, args.dpr)
	write_data2(write_path, args.game, args.dpr, eq_profs, method='multi_m7') # conservative
	#write_data2(write_path, args.game, args.dpr, eq_profs, method='multi_m8') # aggressive
	##eq_profs = three_layer(args.profpath, args.gamepath, args.game, args.dpr)
	##write_data2(args.profpath, args.game, args.dpr, eq_profs, method='multi_m7') # conservative
	##write_data2(args.profpath, args.game, args.dpr, eq_profs, method='multi_m8') # aggressive
	print('Wrote data.')

	#USE
	#print('Method 3:')
	#eq_profs = m1(profpath, gamepath, args.game, args.dpr)
	#write_data2(write_path, args.game, args.dpr, eq_profs, method='m3')
	##print('Method 1:')
	##eq_profs = m1(args.profpath, args.gamepath, args.game, args.dpr)
	##write_data(args.profpath, args.game, args.dpr, eq_profs, method='m1')
	#print('Wrote data.')

	#skip
	#eq_profs = m2(args.profpath, args.gamepath, args.game, args.dpr)
	#write_data(args.profpath, args.game, args.dpr, eq_profs, method='m2')

	#skip
	#print('Method 3:')
	#eq_profs = m3(args.profpath, args.gamepath, args.game, args.dpr)
	#write_data(args.profpath, args.game, args.dpr, eq_profs, method='m3')
	#print('Wrote data.')

	#USE
	#print('Method 4:')
	#eq_profs = m4(profpath, gamepath, args.game, args.dpr)
	#write_data2(write_path, args.game, args.dpr, eq_profs, method='m4')
	##print('Method 4:')
	##eq_profs = m4(args.profpath, args.gamepath, args.game, args.dpr)
	##write_data(args.profpath, args.game, args.dpr, eq_profs, method='m4')
	#print('Wrote data.')

	#USE
	#print('Method 5:')
	#eq_profs = m5(profpath, gamepath, args.game, args.dpr)
	#write_data2(write_path, args.game, args.dpr, eq_profs, method='m5')
	##print('Method 5:')
	##eq_profs = m5(args.profpath, args.gamepath, args.game, args.dpr)
	##write_data(args.profpath, args.game, args.dpr, eq_profs, method='m5')
	#print('Wrote data.')

	#skip
	#print('Method 6:')
	#eq_profs = m6(args.profpath, args.gamepath, args.game, args.dpr)
	#write_data(args.profpath, args.game, args.dpr, eq_profs, method='m6')
	#print('Wrote data.')

	#USE
	#print('Method 6:')
	#eq_profs = m7(profpath, gamepath, args.game, args.dpr)
	#write_data2(write_path, args.game, args.dpr, eq_profs, method='m6')
	##print('Method 7:')
	##eq_profs = m7(args.profpath, args.gamepath, args.game, args.dpr)
	##write_data(args.profpath, args.game, args.dpr, eq_profs, method='m7')
	#print('Wrote data.')
	"""
