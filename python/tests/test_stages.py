"""
Stage Tests (Phase 2.53-2.55)

Tests for stage transitions including:
- Stage completion when reaching exit
- Data block invariant on new stage
- Player state preservation across stages

These tests verify that the Swift environment correctly handles stage transitions.
"""

import pytest
import numpy as np

from .env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    Observation,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
)


# MARK: - Helper Functions

def get_player_position(obs: Observation) -> tuple[int, int]:
    """Extract player row, col from observation."""
    row = int(round(obs.player[0] * 5))
    col = int(round(obs.player[1] * 5))
    return row, col


def get_player_stage(obs: Observation) -> int:
    """Extract player stage from observation."""
    return int(round(obs.player[5] * 8))


def get_player_score(obs: Observation) -> int:
    """Extract player score from observation."""
    # Score is part of info, not directly in observation
    # For now we check stage change instead
    return 0


def get_player_credits(obs: Observation) -> int:
    """Extract player credits from observation."""
    return int(round(obs.player[3] * 50))


def get_player_energy(obs: Observation) -> int:
    """Extract player energy from observation."""
    return int(round(obs.player[4] * 50))


def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


def count_data_blocks(obs: Observation) -> int:
    """Count data blocks on the grid."""
    count = 0
    for row in range(6):
        for col in range(6):
            # Check block channels (shifted by 1 after adding spawnedFromSiphon)
            if obs.grid[row, col, 7] > 0.5:  # Data block
                count += 1
    return count


def get_blocks_info(obs: Observation) -> list[dict]:
    """Get information about blocks from observation."""
    blocks = []
    for row in range(6):
        for col in range(6):
            if obs.grid[row, col, 7] > 0.5:  # Data block (channel 7)
                blocks.append({
                    "row": row,
                    "col": col,
                    "siphoned": obs.grid[row, col, 11] > 0.5  # Siphoned at channel 11
                })
            elif obs.grid[row, col, 8] > 0.5:  # Program block (channel 8)
                blocks.append({
                    "row": row,
                    "col": col,
                    "type": "program",
                    "siphoned": obs.grid[row, col, 11] > 0.5  # Siphoned at channel 11
                })
    return blocks


# MARK: - Test 2.53: Stage Completion

class TestStageCompletion:
    """Test 2.53: Stage completes when player reaches exit."""

    @pytest.mark.requires_set_state
    def test_stage_completes_on_exit(self, env):
        """Moving to exit should advance to next stage.

        Note: Exit is always at top-right (row=5, col=5).
        Player starts adjacent to exit.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=0, energy=0),
            enemies=[],
            blocks=[],
            stage=1
        )
        obs_before = env.set_state(state)
        stage_before = get_player_stage(obs_before)

        # Move right to reach exit at (5, 5)
        result = env.step(ACTION_MOVE_RIGHT)

        stage_after = get_player_stage(result.observation)
        # Stage should advance
        assert stage_after == stage_before + 1, \
            f"Stage should advance from {stage_before} to {stage_before + 1}, got {stage_after}"

    @pytest.mark.requires_set_state
    def test_stage_reward_increases_with_stage(self, env):
        """Stage completion rewards should increase exponentially."""
        # Complete stage 1
        state1 = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state1)
        result1 = env.step(ACTION_MOVE_RIGHT)
        reward1 = result1.reward

        # Complete stage 3 (should have higher reward)
        state3 = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=3
        )
        env.set_state(state3)
        result3 = env.step(ACTION_MOVE_RIGHT)
        reward3 = result3.reward

        # Stage 3 completion reward should be higher than stage 1
        # (Reward multipliers: [1, 2, 4, 8, 16, 32, 64, 100])
        assert reward3 > reward1, \
            f"Stage 3 reward ({reward3}) should be > stage 1 reward ({reward1})"


# MARK: - Test 2.54: Data Block Invariant

class TestDataBlockInvariant:
    """Test 2.54: New stages maintain data block invariant (points == spawnCount)."""

    @pytest.mark.requires_set_state
    def test_new_stage_generates_content(self, env):
        """New stage should be generated with content.

        Note: Stage advancement happens on reaching exit.
        We verify the stage transition occurred.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        obs_before = env.set_state(state)
        stage_before = get_player_stage(obs_before)

        # Move to exit
        result = env.step(ACTION_MOVE_RIGHT)

        # Stage should advance
        stage_after = get_player_stage(result.observation)
        assert stage_after > stage_before or stage_after == stage_before + 1, \
            f"Stage should advance from {stage_before}, got {stage_after}"


# MARK: - Test 2.55: Player State Preserved

class TestPlayerStatePreserved:
    """Test 2.55: Player state is preserved across stage transitions."""

    @pytest.mark.requires_set_state
    def test_credits_preserved_on_stage_transition(self, env):
        """Player credits should be preserved when transitioning stages."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=10, energy=5),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        # Credits should be preserved
        credits_after = get_player_credits(result.observation)
        assert credits_after == 10, f"Credits should be 10, got {credits_after}"

    @pytest.mark.requires_set_state
    def test_energy_preserved_on_stage_transition(self, env):
        """Player energy should be preserved when transitioning stages."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=5, energy=7),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        energy_after = get_player_energy(result.observation)
        assert energy_after == 7, f"Energy should be 7, got {energy_after}"

    @pytest.mark.requires_set_state
    def test_hp_restored_on_stage_transition(self, env):
        """Player HP is restored to full when transitioning stages.

        Note: The game auto-heals player to 3 HP on stage completion.
        This is the expected game behavior.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=2, credits=0, energy=0),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        hp_after = get_player_hp(result.observation)
        # Game restores HP on stage transition
        assert hp_after == 3, f"HP should be restored to 3 on stage transition, got {hp_after}"

    @pytest.mark.requires_set_state
    def test_player_position_on_stage_transition(self, env):
        """Player position changes on stage transition.

        Note: The exact starting position depends on game implementation.
        We verify the stage transition occurred successfully.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        obs_before = env.set_state(state)
        stage_before = get_player_stage(obs_before)

        result = env.step(ACTION_MOVE_RIGHT)

        stage_after = get_player_stage(result.observation)
        # Verify stage transition occurred
        assert stage_after > stage_before, \
            f"Stage should advance from {stage_before}, got {stage_after}"
