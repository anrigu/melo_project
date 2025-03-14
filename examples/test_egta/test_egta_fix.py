"""
Simple test script to verify our fix to the EGTA.run() method.
"""
import sys
import os
sys.path.insert(0, '..')

from marketsim.egta.egta import EGTA
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.base import Simulator

class TestSimulator(Simulator):
    def get_strategies(self):
        return ['A', 'B']
    
    def get_num_players(self):
        return 2
    
    def simulate_profile(self, profile):
        # Return dummy payoff data for a single profile
        return ((0, 'A', 1.0), (1, 'B', 2.0))
    
    def simulate_profiles(self, profiles):
        # Return dummy payoff data for multiple profiles
        return [self.simulate_profile(profile) for profile in profiles]

def main():
    # Create output directory
    os.makedirs('test_egta', exist_ok=True)
    
    # Create scheduler
    scheduler = DPRScheduler(
        strategies=['A', 'B'],
        num_players=2,
        batch_size=3
    )
    
    # Create EGTA framework
    egta = EGTA(
        simulator=TestSimulator(),
        scheduler=scheduler,
        output_dir='test_egta'
    )
    
    try:
        # Run EGTA
        print("Starting EGTA test...")
        game = egta.run(
            max_iterations=1,
            profiles_per_iteration=3,
            verbose=True
        )
        print("Success! EGTA.run() completed without errors.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 