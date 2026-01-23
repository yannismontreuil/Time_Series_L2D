#!/usr/bin/env python3
"""
Test script to verify the flexible expert system implementation.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, '/Users/sky/PycharmProjects/Time_Series_L2D')

def test_flexible_expert_system():
    """Test the flexible expert system."""
    print("Testing flexible expert system implementation...")

    try:
        from environment.etth1_env import ETTh1TimeSeriesEnv

        # Test 1: Enable specific experts by name
        print("\n=== Test 1: Enable specific experts ===")
        enabled_experts = ["ar1_low_var", "ar1_high_var", "nn_early", "ar2_segment1"]
        env = ETTh1TimeSeriesEnv(
            csv_path="data/ETTh1.csv",
            enabled_experts=enabled_experts,
            T=100,
            seed=42
        )

        print(f"✓ Created environment with {env.num_experts} experts")
        print(f"✓ Enabled experts: {env.enabled_experts}")
        print(f"✓ Expert types: {env.expert_types}")

        # Test predictions
        x_t = env.get_context(10)
        for j in range(env.num_experts):
            pred = env.expert_predict(j, x_t)
            expert_type = env.expert_types[j]
            print(f"  Expert {j} ({expert_type}): prediction = {pred:.4f}")

        # Test 2: Backward compatibility with num_experts
        print("\n=== Test 2: Backward compatibility ===")
        env2 = ETTh1TimeSeriesEnv(
            csv_path="data/ETTh1.csv",
            num_experts=3,
            T=100,
            seed=42
        )

        print(f"✓ Created environment with num_experts=3")
        print(f"✓ Auto-enabled experts: {env2.enabled_experts}")
        print(f"✓ Expert types: {env2.expert_types}")

        # Test 3: All expert types
        print("\n=== Test 3: All expert types ===")
        all_experts = ["ar1_low_var", "ar1_high_var", "nn_early", "nn_late", "ar2_segment1", "ar2_segment2"]
        env3 = ETTh1TimeSeriesEnv(
            csv_path="data/ETTh1.csv",
            enabled_experts=all_experts,
            T=200,  # Need more data for AR(2) training
            seed=42
        )

        print(f"✓ Created environment with all {env3.num_experts} expert types")

        x_t = env3.get_context(50)
        for j in range(env3.num_experts):
            pred = env3.expert_predict(j, x_t)
            expert_type = env3.expert_types[j]
            print(f"  Expert {j} ({expert_type}): prediction = {pred:.4f}")

        # Test different predictions for AR(2) experts
        print("\n=== Test 4: AR(2) expert predictions ===")
        ar2_experts = [j for j, etype in env3.expert_types.items() if etype.startswith('ar2_')]
        for j in ar2_experts:
            if j in env3._ar2_params and env3._ar2_params[j] is not None:
                w1, w2, b = env3._ar2_params[j]
                print(f"  AR(2) Expert {j}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

        print("\n✓ All tests passed! Flexible expert system is working correctly.")
        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flexible_expert_system()
    sys.exit(0 if success else 1)
