import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.egta.egta import EGTA
from marketsim.egta.schedulers.dpr import DPRScheduler

def test_dpr_verification():
    """Test DPR equilibrium verification in full game"""
    
    # Setup - Use only mobi agents for a symmetric game
    num_mobi_agents = 10  # Small for testing
    num_zi_agents = 15   # No background agents to avoid RSG complexity
    
    print("Testing DPR Equilibrium Verification")
    print("=" * 50)
    
    # Create simulator
    simulator = MeloSimulator(
        num_strategic=num_mobi_agents,
        sim_time=1000,  # Short for testing
        lam=0.006,
        mean=1000000,
        r=0.05,
        shock_var=100,
        q_max=10,
        pv_var=5000000,
        shade=[10, 30],
        holding_period=290,
        lam_melo=0.001,
        num_zi=num_zi_agents,
        num_hbl=0,
        reps=100,  # Few reps for testing
        #force_rsg=True  # Force symmetric game
    )
    
    strategies = simulator.get_strategies()
    print(f"Strategies: {strategies}")
    print(f"Full game players: {num_mobi_agents}")
    print(f"Is RSG: {simulator.is_role_symmetric()}")
    
    # Create DPR scheduler with reduction
    reduction_size = 4  # Reduce from 7 to 4 players
    scheduler = DPRScheduler(
        strategies=strategies,
        num_players=num_mobi_agents,
        batch_size=10,
        reduction_size=reduction_size,
        seed=42
        
    )
    
    print(f"Reduced game players: {reduction_size}")
    print(f"Scaling factor: {scheduler.scaling_factor:.2f}")
    
    # Run EGTA
    device = torch.device("cpu")
    egta = EGTA(
        simulator=simulator,
        scheduler=scheduler,
        device=device,
        output_dir="results/dpr_verification_test",
        max_profiles=30,  # Small for testing
        seed=42
    )
    
    # Run with explicit regret threshold
    game = egta.run(
        max_iterations=3,
        profiles_per_iteration=10,
        verbose=True,
        quiesce_kwargs={
            'num_iters': 10,
            'num_random_starts': 5,
            'regret_threshold': 0.01,  # Epsilon for verification
            'solver': 'replicator',
            'solver_iters': 1000
        }
    )
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print(f"Final equilibria count: {len(egta.equilibria)}")
    
    if egta.equilibria:
        eq_mix, eq_regret = egta.equilibria[0]
        print(f"Best equilibrium regret: {float(eq_regret):.6f}")
        for i, strat in enumerate(game.strategy_names):
            if eq_mix[i] > 0.01:
                print(f"  {strat}: {eq_mix[i].item():.4f}")

if __name__ == "__main__":
    os.makedirs("results/dpr_verification_test", exist_ok=True)
    test_dpr_verification() 