"""
Test script to inspect observation structure and verify all game state is captured.
"""

from hack_env import HackEnv
import json

# Create environment
env = HackEnv(debug_scenario=True)

# Reset and get initial observation
obs, info = env.reset()

# Get the raw JSON response to see what's actually being sent
response = env._send_command({"action": "reset"})

print("=" * 80)
print("RAW JSON OBSERVATION FROM SWIFT:")
print("=" * 80)
print(json.dumps(response["observation"], indent=2))

print("\n" + "=" * 80)
print("CHECKING BLOCK FEATURES:")
print("=" * 80)

# Check a few cells to see what block information is available
for row_idx, row in enumerate(response["observation"]["cells"]):
    for col_idx, cell in enumerate(row):
        if "block" in cell:
            block = cell["block"]
            print(f"\nCell ({row_idx}, {col_idx}) has block:")
            print(f"  blockType: {block.get('blockType')}")
            print(f"  points: {block.get('points')}")
            print(f"  programType: {block.get('programType')}")  # Is this present?
            print(f"  transmissionSpawnCount: {block.get('transmissionSpawnCount')}")
            print(f"  isSiphoned: {block.get('isSiphoned')}")

print("\n" + "=" * 80)
print("PYTHON NUMPY OBSERVATION SHAPE:")
print("=" * 80)
print(f"player: {obs['player'].shape} = {obs['player']}")
print(f"grid: {obs['grid'].shape}")
print(f"flags: {obs['flags'].shape}")

# Check if programType is being encoded
print("\n" + "=" * 80)
print("SAMPLE GRID FEATURES FOR CELLS WITH BLOCKS:")
print("=" * 80)
for row_idx in range(6):
    for col_idx in range(6):
        cell_features = obs['grid'][row_idx, col_idx, :]
        # Check if this cell has a block (feature index 3)
        if cell_features[3] != 0:  # block_type != 0
            print(f"\nCell ({row_idx}, {col_idx}) grid features:")
            print(f"  Features 0-2 (enemy): {cell_features[0:3]}")
            print(f"  Features 3-5 (block): {cell_features[3:6]}")
            print(f"  Features 6 (transmission): {cell_features[6]}")
            print(f"  Features 7-8 (resources): {cell_features[7:9]}")
            print(f"  Features 9-10 (special): {cell_features[9:11]}")

env.close()
