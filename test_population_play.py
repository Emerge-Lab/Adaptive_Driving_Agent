#!/usr/bin/env python3
"""
Simple test for population play functionality.
"""
import numpy as np
from pufferlib.ocean.drive import Drive

def test_population_play():
    """Test basic population play initialization and step."""
    print("Testing population play initialization...")

    # Create environment with population play enabled
    env = Drive(
        num_agents=128,  # Total agents including ego + co-player
        num_ego_agents=64,  # Number of ego agents to train
        ego_probability=0.5,  # 50/50 split
        num_maps=100,  # Use more maps to avoid empty map selection
        population_play=True,
        action_type="discrete",
        dynamics_model="classic",
        scenario_length=91,
        resample_frequency=-1,  # Disable resampling for this test
        ini_file="pufferlib/config/ocean/population.ini",
    )

    print(f"✓ Environment created successfully")
    print(f"  Total agents: {env.total_agents}")
    print(f"  Ego agents: {len(env.ego_ids)}")
    print(f"  Co-player agents: {len(env.co_player_ids)}")
    print(f"  Observation space for learning: {env.num_agents} (should be ego agents only)")

    # Reset environment
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Expected: ({len(env.ego_ids)}, {env.num_obs})")

    assert obs.shape[0] == len(env.ego_ids), f"Expected {len(env.ego_ids)} observations, got {obs.shape[0]}"

    # Step environment
    # Only provide actions for ego agents
    actions = np.random.randint(0, [7, 13], size=(len(env.ego_ids), 2))

    obs, rewards, terminals, truncations, info = env.step(actions)

    print(f"✓ Step successful")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Terminals shape: {terminals.shape}")

    # Verify shapes
    assert obs.shape[0] == len(env.ego_ids), "Observation shape mismatch"
    assert rewards.shape[0] == len(env.ego_ids), "Rewards shape mismatch"
    assert terminals.shape[0] == len(env.ego_ids), "Terminals shape mismatch"

    # Run a few more steps
    for i in range(5):
        actions = np.random.randint(0, [7, 13], size=(len(env.ego_ids), 2))
        obs, rewards, terminals, truncations, info = env.step(actions)

    print(f"✓ Multiple steps successful")
    print(f"\n✅ Population play test PASSED!")
    return True

def test_self_play():
    """Test that self-play still works (population_play=False)."""
    print("\nTesting self-play mode (baseline)...")

    env = Drive(
        num_agents=64,
        num_maps=1,
        population_play=False,
        action_type="discrete",
        dynamics_model="classic",
        scenario_length=91,
        resample_frequency=-1,
        ini_file="pufferlib/config/ocean/drive.ini",
    )

    print(f"✓ Self-play environment created")
    print(f"  Num agents: {env.num_agents}")

    obs, info = env.reset()
    print(f"✓ Reset successful, obs shape: {obs.shape}")

    actions = np.random.randint(0, [7, 13], size=(env.num_agents, 2))
    obs, rewards, terminals, truncations, info = env.step(actions)

    print(f"✓ Step successful")
    print(f"✅ Self-play test PASSED!")
    return True

if __name__ == "__main__":
    try:
        # Test self-play first to make sure we didn't break anything
        test_self_play()

        # Test population play
        test_population_play()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! Population play is working!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
