"""
Program Tests (Phase 2.16-2.38)

Tests for program mechanics. Each program has:
- Resource costs (credits, energy)
- Applicability conditions
- Primary effects
- Secondary effects (stuns, damage, etc.)

These tests verify that the Swift environment correctly implements program mechanics.
"""

import pytest
import numpy as np

from .env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    Transmission,
    Observation,
    PROGRAM_PUSH,
    PROGRAM_PULL,
    PROGRAM_WAIT,
    PROGRAM_POLY,
    PROGRAM_RESET,
    PROGRAM_SIPH_PLUS,
    ACTION_MOVE_UP,
)


# MARK: - Helper Functions

def get_player_energy(obs: Observation) -> int:
    """Extract player energy from observation."""
    return int(round(obs.player[4] * 50))


def get_player_credits(obs: Observation) -> int:
    """Extract player credits from observation."""
    return int(round(obs.player[3] * 50))


def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


def get_player_siphons(obs: Observation) -> int:
    """Extract player data siphons from observation."""
    return int(round(obs.player[6] * 10))


def get_enemy_at(obs: Observation, row: int, col: int) -> dict | None:
    """Get enemy info at position."""
    if not np.any(obs.grid[row, col, 0:4] > 0):
        return None
    enemy_types = ["virus", "daemon", "glitch", "cryptog"]
    for i, etype in enumerate(enemy_types):
        if obs.grid[row, col, i] > 0:
            hp = int(round(obs.grid[row, col, 4] * 3))
            stunned = obs.grid[row, col, 5] > 0.5
            return {"type": etype, "hp": hp, "stunned": stunned, "row": row, "col": col}
    return None


def find_enemies(obs: Observation) -> list[dict]:
    """Find all enemies on the grid."""
    enemies = []
    for row in range(6):
        for col in range(6):
            enemy = get_enemy_at(obs, row, col)
            if enemy:
                enemies.append(enemy)
    return enemies


def count_enemies(obs: Observation) -> int:
    """Count enemies on grid."""
    return len(find_enemies(obs))


# MARK: - Test 2.16: PUSH Program

class TestPushProgram:
    """Test 2.16: PUSH program pushes enemies away."""

    @pytest.mark.requires_set_state
    def test_push_enemies_away(self, swift_env):
        """PUSH should move enemies away from player and cost energy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, dataSiphons=0),
            enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        obs = swift_env.set_state(state)
        energy_before = get_player_energy(obs)
        assert energy_before == 2, f"Should have 2 energy, got {energy_before}"

        result = swift_env.step(PROGRAM_PUSH)

        # Energy should be consumed (PUSH costs 0C, 2E)
        energy_after = get_player_energy(result.observation)
        assert energy_after == 0, f"Energy should be 0 after PUSH, got {energy_after}"

        # Enemy should be pushed away (row increases)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, f"Should still have 1 enemy, got {len(enemies)}"
        assert enemies[0]["row"] == 5, f"Enemy should be at row 5, got {enemies[0]['row']}"

    @pytest.mark.requires_set_state
    def test_push_requires_enemies(self, swift_env):
        """PUSH should be invalid without enemies."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            enemies=[],  # No enemies
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_PUSH not in valid, f"PUSH should be invalid without enemies, got {valid}"


# MARK: - Test 2.17: PULL Program

class TestPullProgram:
    """Test 2.17: PULL program pulls enemies toward player."""

    @pytest.mark.requires_set_state
    def test_pull_enemies_toward(self, swift_env):
        """PULL should move enemies toward player."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=2),
            enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],
            owned_programs=[PROGRAM_PULL],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_PULL)

        # Enemy should be pulled closer
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 4, f"Enemy should be at row 4, got {enemies[0]['row']}"


# MARK: - Test 2.20: POLY Program

class TestPolyProgram:
    """Test 2.20: POLY program transforms enemies."""

    @pytest.mark.requires_set_state
    def test_poly_randomizes_enemy_types(self, swift_env):
        """POLY should change enemy type."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=1, energy=1),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],
            owned_programs=[PROGRAM_POLY],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_POLY)

        # Enemy type should change (guaranteed different from virus)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["type"] != "virus", f"Enemy type should change, still {enemies[0]['type']}"


# MARK: - Test 2.21: WAIT Program

class TestWaitProgram:
    """Test 2.21: WAIT program ends turn."""

    @pytest.mark.requires_set_state
    def test_wait_ends_turn(self, swift_env):
        """WAIT should end turn and cause enemy movement."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=1),
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Daemon should move closer (1 cell per turn)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        # Daemon was at row 5, should move to row 4
        assert enemies[0]["row"] == 4, f"Daemon should be at row 4, got {enemies[0]['row']}"


# MARK: - Test 2.27: SIPH+ Program

class TestSiphPlusProgram:
    """Test 2.27: SIPH+ program grants a data siphon."""

    @pytest.mark.requires_set_state
    def test_siph_plus_gains_data_siphon(self, swift_env):
        """SIPH+ should give player a data siphon."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0),
            owned_programs=[PROGRAM_SIPH_PLUS],
            enemies=[],
            stage=1
        )
        obs = swift_env.set_state(state)
        siphons_before = get_player_siphons(obs)
        assert siphons_before == 0, f"Should start with 0 siphons, got {siphons_before}"

        result = swift_env.step(PROGRAM_SIPH_PLUS)

        siphons_after = get_player_siphons(result.observation)
        assert siphons_after == 1, f"Should have 1 siphon after SIPH+, got {siphons_after}"

        # Credits should be consumed (5C cost)
        credits_after = get_player_credits(result.observation)
        assert credits_after == 0, f"Credits should be 0 after SIPH+, got {credits_after}"


# MARK: - Test 2.30: RESET Program

class TestResetProgram:
    """Test 2.30: RESET program restores HP."""

    @pytest.mark.requires_set_state
    def test_reset_restores_hp(self, swift_env):
        """RESET should restore player HP to max."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=1, credits=0, energy=4),
            owned_programs=[PROGRAM_RESET],
            enemies=[],
            stage=1
        )
        obs = swift_env.set_state(state)
        hp_before = get_player_hp(obs)
        assert hp_before == 1, f"Should start with 1 HP, got {hp_before}"

        result = swift_env.step(PROGRAM_RESET)

        hp_after = get_player_hp(result.observation)
        assert hp_after == 3, f"HP should be 3 after RESET, got {hp_after}"

    @pytest.mark.requires_set_state
    def test_reset_requires_low_hp(self, swift_env):
        """RESET should be invalid at full HP."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),  # Full HP
            owned_programs=[PROGRAM_RESET],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_RESET not in valid, f"RESET should be invalid at full HP, got {valid}"


# MARK: - Program Chaining Tests

class TestProgramChaining:
    """Test that programs don't end turn (except WAIT)."""

    @pytest.mark.requires_set_state
    def test_program_does_not_end_turn(self, swift_env):
        """Programs (except WAIT) should not end the turn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0),
            owned_programs=[PROGRAM_SIPH_PLUS],
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_SIPH_PLUS)

        # Enemy should NOT have moved (turn didn't end)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 5, f"Enemy should still be at row 5, got {enemies[0]['row']}"

        # Movement should still be valid (turn not ended)
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Movement should be valid after program, got {valid}"
