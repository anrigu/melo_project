from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch as th
from fourheap.constants import BUY, SELL
from simulator.sampled_arrival_simulator import SimulatorSampledArrival
from wrappers.SP_wrapper import SPEnv
from wrappers.Paired_SP_wrapper import NonSPEnv
from private_values.private_values import PrivateValues
import torch.distributions as dist
import torch
from fundamental.mean_reverting import GaussianMeanReverting
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC, DDPG, TD3, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from custom_callback import SaveOnBestTrainingRewardCallback
from wrappers.CustomBuffer import WeightedReplayBuffer
import torch

SIM_TIME = 10000
TOTAL_ITERS = 10000
NUM_AGENTS = 25
LEARNING = True
LEARNING_ACTIONS = True
graphVals = 200
printVals = 100

valueAgentsSpoof = []
valueAgentsNon = []
diffs = []
env_trades = []
sim_trades = []
spoof_activity = []
sell_above_best_avg = []
buy_below_best_avg = []
spoofer_position = []
spoof_mid_prices = []
nonspoof_mid_prices = []
nonspoofer_position = []

path = "spoofer_exps/PPO_options/16_RPPO/b"
CALLBACK_LOG_DIR = "spoofer_exps/PPO_options/16_RPPO/c"

print("GRAPH SAVE PATH", path)
print("CALLBACK PATH", CALLBACK_LOG_DIR)

normalizers = {"fundamental": 1e5, "pv":5e3, "reward":1e2, "invt": 10, "cash": 1e5}
# normalizers = {"fundamental": 1, "reward":1, "min_order_val": 1, "invt": 1, "cash": 1}

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

def make_env(spEnv: SPEnv):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init() -> SPEnv:
        env = spEnv
        env.reset()
        return spEnv

    return _init

def run():
    if LEARNING:
        print(torch.cuda.is_available())
        learningEnv = SPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=5e-3,
                    lamSP=2e-2,
                    mean=1e5,
                    r=0.05,
                    shock_var=1e4,
                    q_max=10,
                    pv_var=5e6,
                    shade=[250,500],
                    normalizers=normalizers,
                    learning = LEARNING,
                    learnedActions=LEARNING_ACTIONS,
                    analytics = True)

        num_cpu = 1  # Number of processes to use
        # Create the vectorized environment
        if num_cpu == 1:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=1, monitor_dir=CALLBACK_LOG_DIR, vec_env_cls=DummyVecEnv)
        else:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=num_cpu, monitor_dir=CALLBACK_LOG_DIR, vec_env_cls=SubprocVecEnv)
        # spEnv = SubprocVecEnv([make_env(env) for _ in range(num_cpu)])

        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=CALLBACK_LOG_DIR)
        # We collect 4 transitions per call to `ènv.step()`
        # and performs 2 gradient steps per call to `ènv.step()`
        # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
        # n_actions = spEnv.action_space.shape[-1]
        
        # model = SAC("MlpPolicy", spEnv, verbose=1, device="cuda")
        # model = TD3("MlpPolicy", spEnv, action_noise=action_noise, replay_buffer_class=WeightedReplayBuffer, replay_buffer_kwargs={"unusual_sample_factor":0.95}, verbose=1, device="cuda")
        model = RecurrentPPO("MlpLstmPolicy", spEnv, verbose=1, device="cuda")
        # action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2))
        # model = PPO("MlpPolicy", spEnv, gamma=0.995, verbose=1, device="cuda")
        # policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[512,512]))
        model.learn(total_timesteps=1e5, progress_bar=True, callback=callback)
        # input(callback.cumulative_window_rewards)
        print("Loading best model...")
        model = PPO.load(os.path.join(CALLBACK_LOG_DIR, "best_model.zip"))
    random.seed(10)
    for i in tqdm(range(TOTAL_ITERS)):
        random_seed = [random.randint(0,100000) for _ in range(10000)]

        a = [PrivateValues(10,5e6) for _ in range(0,NUM_AGENTS + 1)]
        sampled_arr = sample_arrivals(5e-3,SIM_TIME)
        spoofer_arrivals = sample_arrivals(3e-2,SIM_TIME)
        fundamental = GaussianMeanReverting(mean=1e5, final_time=SIM_TIME + 1, r=0.05, shock_var=1e4)
        random.seed(12)
        if LEARNING_ACTIONS:
            sim = SPEnv(num_background_agents=NUM_AGENTS,
                sim_time=SIM_TIME,
                lam=5e-3,
                lamSP=3e-2,
                mean=1e5,
                r=0.05,
                shock_var=1e4,
                q_max=10,
                pv_var=5e6,
                shade=[250,500],
                normalizers=normalizers,
                pvalues = a,
                sampled_arr=sampled_arr,
                spoofer_arrivals=spoofer_arrivals,
                fundamental = fundamental,
                learning = False,
                learnedActions = False,
                analytics = True,
                random_seed = random_seed
                )
        else:
            sim = NonSPEnv(num_background_agents=NUM_AGENTS,
                sim_time=SIM_TIME,
                lam=5e-3,
                lamSP=3e-2,
                mean=1e5,
                r=0.05,
                shock_var=1e4,
                q_max=10,
                pv_var=5e6,
                shade=[250,500],
                pvalues = a,
                sampled_arr=sampled_arr,
                spoofer_arrivals=spoofer_arrivals,
                fundamental = fundamental,
                analytics=True,
                random_seed = random_seed
                )
        observation, info = sim.reset()
        random.seed(12)
        env = SPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=5e-3,
                    lamSP=3e-2,
                    mean=1e5,
                    r=0.05,
                    shock_var=1e4,
                    q_max=10,
                    pv_var=5e6,
                    shade=[250,500],
                    normalizers=normalizers,
                    pvalues = a,
                    sampled_arr=sampled_arr,
                    spoofer_arrivals=spoofer_arrivals,
                    fundamental = fundamental,
                    learning = False,
                    learnedActions = LEARNING_ACTIONS,
                    analytics = True,
                    random_seed = random_seed
                    )

        observation, info = env.reset()

        random.seed(8)
        while sim.time < SIM_TIME:
            if LEARNING_ACTIONS:
                action = sim.action_space.sample()  # this is where you would insert your policy
                observation, reward, terminated, truncated, info = sim.step(action)
            else:
                sim.step()

        random.seed(8)
        while env.time < SIM_TIME:
            if LEARNING:
                action, _states = model.predict(observation, deterministic=True)
            else:
                action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

        def estimate_fundamental(t):
            mean = 1e5
            r = 0.05
            T = 10000
            val = sim.markets[0].fundamental.get_value_at(t)
            rho = (1 - r) ** (T - t)

            estimate = (1 - rho) * mean + rho * val
            return estimate

        diffs.append(np.subtract(np.array(list(env.most_recent_trade.values())),np.array(list(sim.most_recent_trade.values()))))
        env_trades.append(list(env.most_recent_trade.values()))
        sim_trades.append(list(sim.most_recent_trade.values()))
        spoof_activity.append(list(env.spoof_activity.values()))

        above_ask_pad = np.full(600, np.nan)
        above_ask_pad[:len(env.sell_above_best)] = env.sell_above_best
        
        below_buy_pad = np.full(600, np.nan)
        below_buy_pad[:len(env.buy_below_best)] = env.buy_below_best
    
        sell_above_best_avg.append(above_ask_pad)
        buy_below_best_avg.append(below_buy_pad)
        spoofer_position.append(list(env.spoofer_quantity.values()))
        nonspoofer_position.append(list(sim.spoofer_quantity.values()))
        fundamental_val = sim.markets[0].get_final_fundamental()
        valuesSpoof = []
        valuesNon = []
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
            valuesSpoof.append(value)
        agent = env.spoofer
        value = agent.position * fundamental_val + agent.cash
        valuesSpoof.append(value)

        for agent_id in sim.agents:
            agent = sim.agents[agent_id]
            value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
            # print(agent.cash, fundamental_val, agent.position, agent.get_pos_value(), value)
            # input()
            valuesNon.append(value)
        #placeholder for additional spoofer
        agent = sim.spoofer
        value = agent.position * fundamental_val + agent.cash
        valuesNon.append(value)
        
        valueAgentsSpoof.append(valuesSpoof)
        valueAgentsNon.append(valuesNon)

        spoof_mid_prices.append(list(env.mid_prices.values()))
        nonspoof_mid_prices.append(list(sim.mid_prices.values()))

        if i % graphVals == 0 or i == TOTAL_ITERS - 1:        
            x_axis = [i for i in range(0, SIM_TIME+1)]

            plt.figure()
            plt.plot(x_axis, np.nanmean(diffs,axis=0))
            plt.title('Spoofer diff - RUNNING AVERAGE')
            plt.xlabel('Timesteps')
            plt.ylabel('Difference')

            # Save the figure
            plt.savefig(path + '/{}_spoofer_diff_sim.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, np.nanmean(env_trades, axis=0), label="spoof")
            plt.plot(x_axis, np.nanmean(sim_trades, axis=0), linestyle='dotted',label="Nonspoof")
            plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Last matched order price')
            plt.title('Spoof v Nonspoof last matched trade price - AVERAGED')
            plt.savefig(path + '/{}_AVG_matched_order_price.png'.format(i))
            plt.close()
            
            plt.figure()
            plt.plot(x_axis, list(env.most_recent_trade.values()), label="spoof", linestyle="dotted")
            plt.plot(x_axis, list(sim.most_recent_trade.values()), linestyle='--',label="Nonspoof")
            plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Last matched order price')
            plt.title('Spoof v Nonspoof last matched trade price - NOT AVERAGED')
            plt.savefig(path + '/{}_NONAVG_matched_order_price.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, np.nanmean(spoofer_position, axis=0), label="spoof")
            plt.plot(x_axis, np.nanmean(nonspoofer_position, axis=0), label="nonspoof")
            plt.xlabel('Timesteps')
            plt.ylabel('Position')
            plt.title('AVERAGED - Position of Spoofer Over Time')
            plt.legend()
            plt.savefig(path + '/{}_AVG_spoofer_position.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, np.nanmean(spoof_activity, axis=0), label="Surplus")
            plt.xlabel('Timesteps')
            plt.ylabel('Surplus')
            plt.title('AVERAGED - Surplus of Spoofer Over Time')
            plt.savefig(path + '/{}_AVG_spoofer_surplus_track.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, list(env.spoof_activity.values()), label="Surplus")
            plt.xlabel('Timesteps')
            plt.ylabel('Surplus')
            plt.title('Not AVG - Surplus of Spoofer Over Time')
            plt.savefig(path + '/{}_NONAVG_spoofer_surplus_track.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, list(env.spoofer_quantity.values()), label="Position")
            plt.xlabel('Timesteps')
            plt.ylabel('Position')
            plt.title('NOTAVERAGED - Surplus of Spoofer Over Time')
            plt.savefig(path + '/{}_NONAVG_spoofer_position.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, np.nanmean(spoofer_position, axis=0), label="Position")
            plt.plot(x_axis, np.nanmean(nonspoofer_position, axis=0), label="Position_Non")
            plt.xlabel('Timesteps')
            plt.ylabel('Position')
            plt.title('AVERAGED - Position of Spoofer Over Time')
            plt.savefig(path + '/{}_AVG_spoofer_position.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(x_axis, np.nanmean(spoof_mid_prices, axis=0), label="Spoof")
            plt.plot(x_axis, np.nanmean(nonspoof_mid_prices, axis=0), label="Nonspoof")
            plt.xlabel('Timesteps')
            plt.ylabel('Midprice')
            plt.legend()
            plt.title('AVERAGED - Midprice Spoof v Nonspoof')
            plt.savefig(path + '/{}_AVG_midprice.png'.format(i))
            plt.close()


            plt.figure()
            plt.hist(range(len(list(env.trade_volume.values()))), bins=len(list(env.trade_volume.values()))//100, weights=list(env.trade_volume.values()), edgecolor='black')
            plt.xlabel('Timesteps')
            plt.ylabel('# trades')
            plt.title('Spoof trade volume')
            plt.savefig(path + '/{}_NONAVG_trade_volume_spoof.png'.format(i))
            plt.close()
            
            plt.figure()
            plt.hist(range(len(list(sim.trade_volume.values()))), bins=len(list(sim.trade_volume.values()))//100, weights=list(sim.trade_volume.values()), edgecolor='black')
            plt.xlabel('Timesteps')
            plt.ylabel('# trades')
            plt.title('Nonspoof trade volume')
            plt.savefig(path + '/{}_NONAVG_trade_volume_nonspoof.png'.format(i))
            plt.close()


            plt.figure()
            a = list(env.spoof_orders.values())
            b = list(env.sell_orders.values())
            plt.plot(x_axis, list(env.spoof_orders.values()), label="spoof", color="magenta", zorder=10)
            plt.plot(x_axis, list(env.best_buys.values()), label="best buys", linestyle="--", color="cyan")
            plt.plot(x_axis, list(env.best_asks.values()), label="best asks", linestyle="--", color="yellow")
            # plt.plot(x_axis, fundamentalEvol, label="fundamental", linestyle="dotted", zorder=0)
            plt.plot(x_axis, list(env.sell_orders.values()), label="sell orders", color="black")
            plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Price')
            plt.title('Price comparisons of spoofer orders - NOT AVERAGED')
            plt.savefig(path + '/{}_spoofer_orders.png'.format(i))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(sell_above_best_avg[0])), np.nanmean(sell_above_best_avg, axis=0), label="sells")
            plt.xlabel('Times of entry')
            plt.ylabel('Spoof sell - best ask')
            plt.title('Spoof sell - best ask - AVERAGED')
            plt.savefig(path + '/{}_AVG_best_ask_diff.png'.format(i))

            plt.figure()
            plt.plot(np.arange(len(buy_below_best_avg[0])), -np.nanmean(buy_below_best_avg, axis=0), label="buys")
            plt.xlabel('Times of entry')
            plt.ylabel('order - best_buy')
            plt.title('Order - best_buy - AVERAGED')
            plt.savefig(path + '/{}_AVG_best_buy_diff.png'.format(i))


            plt.figure()
            bar_width = 0.35
            num_agents = [j for j in range(NUM_AGENTS + 1)]
            num_agent_non= [x + bar_width for x in num_agents]
            plotSpoof = np.nanmean(valueAgentsSpoof, axis = 0)
            plotNon = np.nanmean(valueAgentsNon, axis = 0)
            plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
            plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
            plt.legend()
            # for bar, values in zip(barsSpoof, plotSpoof):
            #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')
            # for bar, values in zip(barsNon, plotNon):
            #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')        
            plt.title('Surplus Comparison Spoof/Nonspoof')
            plt.xlabel('Agent')
            plt.ylabel('Values')
            plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
            plt.savefig(path + '/{}_surpluses_sim.png'.format(i))
            plt.close()

        if i % printVals == 0 or i == TOTAL_ITERS - 1:
            print("SPOOFER")
            print(plotSpoof)
            print("NONSPOOFER")
            print(plotNon)
            print("\n SPOOFER POSITION TRACK")
            print(env.spoofer.position)

if __name__ == "__main__":
    run()