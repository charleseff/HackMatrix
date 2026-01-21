"""
Action Mask Tests (Phase 2.56-2.63)

Tests for action masking including:
- Movement masked by walls/edges
- Movement masked by blocks
- Siphon validity based on data siphons
- Programs masked when not owned
- Programs masked when insufficient credits/energy
- Programs masked when applicability conditions not met
- Mask updates after state changes

These tests verify that the Swift environment correctly implements action masking.
"""

import pytest
import numpy as np

from ..env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    Observation,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SIPHON,
    PROGRAM_PUSH,
    PROGRAM_CRASH,
    PROGRAM_D_BOM,
    PROGRAM_ANTI_V,
    PROGRAM_SHOW,
    PROGRAM_RESET,
    PROGRAM_UNDO,
)


# MARK: - Test 2.56: Movement Masked by Edges

class TestMovementMaskedByEdges:
    """Test 2.56: Movement is masked at grid edges."""

    @pytest.mark.requires_set_state
    def test_movement_masked_at_bottom_left(self, env):
        """At (0,0), down and left should be masked."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert ACTION_MOVE_DOWN not in valid, f"Down should be masked at row 0, got {valid}"
        assert ACTION_MOVE_LEFT not in valid, f"Left should be masked at col 0, got {valid}"
        assert ACTION_MOVE_UP in valid, "Up should be valid at row 0"
        assert ACTION_MOVE_RIGHT in valid, "Right should be valid at col 0"

    @pytest.mark.requires_set_state
    def test_movement_masked_at_top_right(self, env):
        """At (5,5), up and right should be masked."""
        state = GameState(
            player=PlayerState(row=5, col=5, hp=3),
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert ACTION_MOVE_UP not in valid, f"Up should be masked at row 5, got {valid}"
        assert ACTION_MOVE_RIGHT not in valid, f"Right should be masked at col 5, got {valid}"
        assert ACTION_MOVE_DOWN in valid, "Down should be valid at row 5"
        assert ACTION_MOVE_LEFT in valid, "Left should be valid at col 5"


# MARK: - Test 2.57: Movement Masked by Blocks

class TestMovementMaskedByBlocks:
    """Test 2.57: Movement is masked by blocks."""

    @pytest.mark.requires_set_state
    def test_movement_blocked_by_adjacent_block(self, env):
        """Movement toward an adjacent block should be masked."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5)],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert ACTION_MOVE_UP not in valid, f"Up should be masked (block at row 4), got {valid}"
        assert ACTION_MOVE_DOWN in valid, "Down should be valid"
        assert ACTION_MOVE_LEFT in valid, "Left should be valid"
        assert ACTION_MOVE_RIGHT in valid, "Right should be valid"


# MARK: - Test 2.58: Siphon Validity

class TestSiphonValidity:
    """Test 2.58: Siphon validity based on data siphons."""

    @pytest.mark.requires_set_state
    def test_siphon_valid_with_siphons(self, env):
        """Siphon should be valid when player has data siphons."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert ACTION_SIPHON in valid, f"Siphon should be valid with siphons, got {valid}"

    @pytest.mark.requires_set_state
    def test_siphon_invalid_without_siphons(self, env):
        """Siphon should be invalid without data siphons."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=0),
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert ACTION_SIPHON not in valid, f"Siphon should be invalid without siphons, got {valid}"


# MARK: - Test 2.59: Programs Masked When Not Owned

class TestProgramsMaskedWhenNotOwned:
    """Test 2.59: Programs are masked when not owned."""

    @pytest.mark.requires_set_state
    def test_programs_masked_when_not_owned(self, env):
        """Programs should be masked when not in owned_programs."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            owned_programs=[],  # No programs owned
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_PUSH not in valid, f"PUSH should be masked (not owned), got {valid}"
        assert PROGRAM_CRASH not in valid, f"CRASH should be masked (not owned), got {valid}"


# MARK: - Test 2.60: Programs Masked When Insufficient Credits

class TestProgramsMaskedInsufficientCredits:
    """Test 2.60: Programs are masked when insufficient credits."""

    @pytest.mark.requires_set_state
    def test_crash_masked_insufficient_credits(self, env):
        """CRASH (3C cost) should be masked with 0 credits."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=10),
            enemies=[Enemy(type="virus", row=4, col=3, hp=2)],  # Adjacent for applicability
            owned_programs=[PROGRAM_CRASH],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_CRASH not in valid, f"CRASH should be masked (0 credits, needs 3), got {valid}"


# MARK: - Test 2.61: Programs Masked When Insufficient Energy

class TestProgramsMaskedInsufficientEnergy:
    """Test 2.61: Programs are masked when insufficient energy."""

    @pytest.mark.requires_set_state
    def test_push_masked_insufficient_energy(self, env):
        """PUSH (2E cost) should be masked with 0 energy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=0),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_PUSH not in valid, f"PUSH should be masked (0 energy, needs 2), got {valid}"


# MARK: - Test 2.62: Programs Masked by Applicability Conditions

class TestProgramsMaskedByApplicability:
    """Test 2.62: Programs are masked when conditions not met."""

    @pytest.mark.requires_set_state
    def test_push_masked_no_enemies(self, env):
        """PUSH should be masked without enemies."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            enemies=[],  # No enemies
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_PUSH not in valid, f"PUSH should be masked (no enemies), got {valid}"

    @pytest.mark.requires_set_state
    def test_d_bom_masked_no_daemon(self, env):
        """D_BOM should be masked without daemon enemy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Not a daemon
            owned_programs=[PROGRAM_D_BOM],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_D_BOM not in valid, f"D_BOM should be masked (no daemon), got {valid}"

    @pytest.mark.requires_set_state
    def test_antiv_masked_no_virus(self, env):
        """ANTI-V should be masked without virus enemy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            enemies=[Enemy(type="daemon", row=5, col=5, hp=3)],  # Not a virus
            owned_programs=[PROGRAM_ANTI_V],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_ANTI_V not in valid, f"ANTI-V should be masked (no virus), got {valid}"

    @pytest.mark.requires_set_state
    def test_show_masked_already_activated(self, env):
        """SHOW should be masked when already activated."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            showActivated=True,
            owned_programs=[PROGRAM_SHOW],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_SHOW not in valid, f"SHOW should be masked (already activated), got {valid}"

    @pytest.mark.requires_set_state
    def test_reset_masked_full_hp(self, env):
        """RESET should be masked at full HP."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),  # Full HP
            owned_programs=[PROGRAM_RESET],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_RESET not in valid, f"RESET should be masked (full HP), got {valid}"

    @pytest.mark.requires_set_state
    def test_undo_masked_empty_history(self, env):
        """UNDO should be masked when game history is empty.

        Note: Fresh state after set_state has no history.
        """
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            owned_programs=[PROGRAM_UNDO],
            stage=1
        )
        env.set_state(state)

        valid = env.get_valid_actions()
        assert PROGRAM_UNDO not in valid, f"UNDO should be masked (no history), got {valid}"


# MARK: - Test 2.63: Mask Updates After State Changes

class TestMaskUpdatesAfterChanges:
    """Test 2.63: Mask updates correctly after state changes."""

    @pytest.mark.requires_set_state
    def test_push_becomes_valid_after_enemy_appears(self, env):
        """PUSH should become valid after an enemy is present."""
        # Initially no enemies
        state1 = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            enemies=[],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        env.set_state(state1)
        valid1 = env.get_valid_actions()
        assert PROGRAM_PUSH not in valid1, "PUSH should be masked initially"

        # Now add an enemy
        state2 = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        env.set_state(state2)
        valid2 = env.get_valid_actions()
        assert PROGRAM_PUSH in valid2, f"PUSH should be valid with enemy, got {valid2}"

    @pytest.mark.requires_set_state
    def test_reset_becomes_valid_after_damage(self, env):
        """RESET should become valid after player takes damage."""
        # Initially full HP
        state1 = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            owned_programs=[PROGRAM_RESET],
            stage=1
        )
        env.set_state(state1)
        valid1 = env.get_valid_actions()
        assert PROGRAM_RESET not in valid1, "RESET should be masked at full HP"

        # Now damaged
        state2 = GameState(
            player=PlayerState(row=3, col=3, hp=1, energy=10),
            owned_programs=[PROGRAM_RESET],
            stage=1
        )
        env.set_state(state2)
        valid2 = env.get_valid_actions()
        assert PROGRAM_RESET in valid2, f"RESET should be valid at low HP, got {valid2}"
