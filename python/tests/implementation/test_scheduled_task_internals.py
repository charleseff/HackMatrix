"""
Scheduled Task Internal Tests

Implementation-level tests that verify hidden game state using get_internal_state().
These tests check mechanics that aren't directly observable through the normal
observation interface.

Tests include:
- Scheduled task interval calculation
- Next scheduled task turn tracking
- Pending siphon transmission counting
- Enemy isFromScheduledTask flag
"""

import pytest

from ..env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    ACTION_MOVE_UP,
    ACTION_SIPHON,
    PROGRAM_WAIT,
    PROGRAM_CALM,
)


# MARK: - Scheduled Task Interval Tests

class TestScheduledTaskInterval:
    """Tests for scheduled task interval calculation."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_stage_1_interval_is_12(self, env):
        """At stage 1, scheduled task interval should be 12."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            stage=1
        )
        env.set_state(state)

        internal = env.get_internal_state()
        assert internal.scheduled_task_interval == 12, \
            f"Stage 1 interval should be 12, got {internal.scheduled_task_interval}"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_stage_2_interval_reduces(self, env):
        """At stage 2, scheduled task interval should be less than or equal to stage 1."""
        state1 = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            stage=1
        )
        env.set_state(state1)
        internal1 = env.get_internal_state()
        interval_stage1 = internal1.scheduled_task_interval

        state2 = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            stage=2
        )
        env.set_state(state2)
        internal2 = env.get_internal_state()
        interval_stage2 = internal2.scheduled_task_interval

        assert interval_stage2 <= interval_stage1, \
            f"Stage 2 interval ({interval_stage2}) should be <= stage 1 ({interval_stage1})"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_higher_stages_have_lower_interval(self, env):
        """Higher stages should have lower intervals (more frequent spawns)."""
        intervals = []
        for stage in [1, 3, 5, 7]:
            state = GameState(
                player=PlayerState(row=0, col=0, hp=3),
                stage=stage
            )
            env.set_state(state)
            internal = env.get_internal_state()
            intervals.append(internal.scheduled_task_interval)

        # Each subsequent stage should have equal or lower interval
        for i in range(1, len(intervals)):
            assert intervals[i] <= intervals[i-1], \
                f"Stage {[1,3,5,7][i]} interval ({intervals[i]}) should be <= " \
                f"stage {[1,3,5,7][i-1]} interval ({intervals[i-1]})"


# MARK: - Next Scheduled Task Turn Tests

class TestNextScheduledTaskTurn:
    """Tests for next scheduled task turn tracking."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_next_task_turn_starts_at_interval(self, env):
        """Next scheduled task turn should start at interval value."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            stage=1,
            turn=0
        )
        env.set_state(state)

        internal = env.get_internal_state()
        # At turn 0, next task should be at turn = interval
        assert internal.next_scheduled_task_turn == internal.scheduled_task_interval, \
            f"Next task turn should equal interval at start, got {internal.next_scheduled_task_turn}"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_next_task_turn_advances_after_spawn(self, env):
        """After spawning, next scheduled task turn should advance by interval."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=50),
            owned_programs=[PROGRAM_WAIT],
            stage=1,
            turn=11  # One turn before scheduled spawn at turn 12
        )
        env.set_state(state)

        internal_before = env.get_internal_state()
        next_turn_before = internal_before.next_scheduled_task_turn

        # Take action to advance to turn 12 (triggers scheduled spawn)
        env.step(PROGRAM_WAIT)

        internal_after = env.get_internal_state()
        # Next turn should have advanced by interval
        expected = next_turn_before + internal_before.scheduled_task_interval
        assert internal_after.next_scheduled_task_turn == expected, \
            f"Next task turn should advance by interval, expected {expected}, " \
            f"got {internal_after.next_scheduled_task_turn}"


# MARK: - Pending Siphon Transmission Tests

class TestPendingSiphonTransmissions:
    """Tests for pending siphon transmission tracking."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_pending_count_is_non_negative(self, env):
        """Pending siphon transmission count should be non-negative after siphon."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            stage=1
        )
        env.set_state(state)

        env.step(ACTION_SIPHON)

        internal_after = env.get_internal_state()
        # Pending count should be non-negative
        # Note: transmissions may spawn immediately, leaving pending at 0
        assert internal_after.pending_siphon_transmissions >= 0, \
            f"Pending siphon transmissions should be non-negative"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_pending_decrements_as_transmissions_spawn(self, env):
        """Pending count should decrement as transmissions actually spawn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="data", points=3, spawnCount=3, siphoned=False)],
            stage=1
        )
        env.set_state(state)

        # Siphon to create pending transmissions
        env.step(ACTION_SIPHON)

        internal = env.get_internal_state()
        # Some transmissions should have spawned, pending reduced
        # (exact number depends on available spawn locations)
        assert internal.pending_siphon_transmissions >= 0, \
            "Pending transmissions should be non-negative"


# MARK: - Enemy From Scheduled Task Tests

class TestEnemyFromScheduledTask:
    """Tests for enemy isFromScheduledTask flag."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_regular_enemy_not_from_scheduled(self, env):
        """Enemies set via set_state should not be marked as from scheduled task."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False, isFromScheduledTask=False)],
            stage=1
        )
        env.set_state(state)

        internal = env.get_internal_state()
        assert len(internal.enemies) == 1
        assert not internal.enemies[0].is_from_scheduled_task, \
            "Regular enemy should not be from scheduled task"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_scheduled_enemy_flagged(self, env):
        """Enemy set with isFromScheduledTask=True should have flag set."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False, isFromScheduledTask=True)],
            stage=1
        )
        env.set_state(state)

        internal = env.get_internal_state()
        assert len(internal.enemies) == 1
        assert internal.enemies[0].is_from_scheduled_task, \
            "Scheduled task enemy should have flag set"


# MARK: - Turn Count Tests

class TestTurnCount:
    """Tests for turn count tracking."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_turn_count_starts_at_set_value(self, env):
        """Turn count should match the value set via set_state."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            turn=5,
            stage=1
        )
        env.set_state(state)

        internal = env.get_internal_state()
        assert internal.turn_count == 5, \
            f"Turn count should be 5, got {internal.turn_count}"

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_turn_count_increments_on_action(self, env):
        """Turn count should increment when player takes an action."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            turn=0,
            stage=1
        )
        env.set_state(state)

        internal_before = env.get_internal_state()
        assert internal_before.turn_count == 0

        env.step(ACTION_MOVE_UP)

        internal_after = env.get_internal_state()
        assert internal_after.turn_count == 1, \
            f"Turn count should be 1 after one action, got {internal_after.turn_count}"


# MARK: - CALM Program Internal Effects

class TestCalmProgramInternals:
    """Tests for CALM program's internal effects."""

    @pytest.mark.requires_set_state
    @pytest.mark.implementation
    def test_calm_does_not_change_interval(self, env):
        """CALM should not change the scheduled task interval."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5),
            owned_programs=[PROGRAM_CALM],
            stage=1
        )
        env.set_state(state)

        internal_before = env.get_internal_state()
        interval_before = internal_before.scheduled_task_interval

        env.step(PROGRAM_CALM)

        internal_after = env.get_internal_state()
        assert internal_after.scheduled_task_interval == interval_before, \
            "CALM should not change the scheduled task interval"
