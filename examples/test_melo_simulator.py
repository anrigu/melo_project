"""
Simple test script to verify that the updated MeloSimulator class works correctly.
"""
from marketsim.egta.simulators.melo_wrapper import MeloSimulator

def test_simulator():
    """Test the MeloSimulator class."""
    print("Creating MeloSimulator instance...")
    simulator = MeloSimulator(
        num_players=10,
        sim_time=1000,
        lam=0.1,
        reps=2  # Use fewer repetitions for quicker testing
    )
    
    print(f"Number of players: {simulator.get_num_players()}")
    
    strategies = simulator.get_strategies()
    print(f"Available strategies: {strategies}")
    
    print(f"Strategy parameters:")
    for strategy in strategies:
        params = simulator.strategy_params[strategy]
        print(f"  {strategy}: CDA={params['cda_proportion']:.2f}, MELO={params['melo_proportion']:.2f}")
    
    print("\nSimulating a simple profile (this may take some time)...")
    profile = [strategies[0]] * simulator.get_num_players()
    try:
        results = simulator.simulate_profile(profile)
        print(f"Simulation completed successfully!")
        print(f"First agent's result: {results[0]}")
    except Exception as e:
        print(f"Error simulating profile: {e}")
        
    print("\nTest completed.")

if __name__ == "__main__":
    test_simulator() 