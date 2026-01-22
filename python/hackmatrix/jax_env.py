"""
Pure functional JAX environment for HackMatrix.
Designed for JIT compilation and TPU training with PureJaxRL.

This is the main environment file that provides reset(), step(), and get_valid_actions().
Game logic is implemented in separate modules for maintainability.
"""

import jax
import jax.numpy as jnp

from .jax_state import (
    EnvState,
    Player,
    create_empty_state,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_PROGRAMS,
    GRID_FEATURES,
    PLAYER_STATE_SIZE,
    PLAYER_MAX_HP,
    EXIT_ROW,
    EXIT_COL,
    DEFAULT_SCHEDULED_TASK_INTERVAL,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SIPHON,
    ACTION_PROGRAM_START,
    DIRECTION_OFFSETS,
    BLOCK_EMPTY,
    PROGRAM_WAIT,
    PROGRAM_COSTS,
)
from .jax_observation import Observation, get_observation


# =============================================================================
# Core Environment Functions
# =============================================================================


def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """
    Initialize environment state.

    Args:
        key: JAX PRNG key for random stage generation

    Returns:
        (initial_state, initial_observation)
    """
    state = create_empty_state(key)

    # Set initial player position at (0, 0)
    player = Player(
        row=jnp.int32(0),
        col=jnp.int32(0),
        hp=jnp.int32(PLAYER_MAX_HP),
        credits=jnp.int32(0),
        energy=jnp.int32(0),
        data_siphons=jnp.int32(0),
        attack_damage=jnp.int32(1),
        score=jnp.int32(0),
    )
    state = state.replace(
        player=player,
        prev_hp=jnp.int32(PLAYER_MAX_HP),
    )

    # TODO: Generate stage 1 with blocks, enemies, resources
    # For now, just return basic state with exit set

    obs = get_observation(state)
    return state, obs


def step(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, Observation, jnp.float32, jnp.bool_]:
    """
    Take one environment step.

    Args:
        state: Current environment state
        action: Action index (0-27)
        key: JAX PRNG key

    Returns:
        (new_state, observation, reward, done)

    Action indices:
        0-3: Movement (up, down, left, right)
        4: Siphon
        5-27: Programs (23 total)
    """
    # Update RNG key
    key, subkey = jax.random.split(key)
    state = state.replace(rng_key=key)

    # Save previous state for reward calculation and UNDO
    state = _save_previous_state(state)

    # Execute action
    state, turn_ends = _execute_action(state, action, subkey)

    # If turn ends, process enemy turn
    state = jax.lax.cond(
        turn_ends,
        lambda s: _process_enemy_turn(s),
        lambda s: s,
        state,
    )

    # Check game end conditions
    player_died = state.player.hp <= 0
    reached_exit = (state.player.row == EXIT_ROW) & (state.player.col == EXIT_COL)
    stage_complete = reached_exit & (state.stage <= 8)

    # Advance stage if complete
    state = jax.lax.cond(
        stage_complete,
        lambda s: _advance_stage(s),
        lambda s: s,
        state,
    )

    # Game is done if player died or won (stage 9)
    game_won = state.stage > 8
    done = player_died | game_won

    # Calculate reward
    reward = _calculate_reward(state, stage_complete, game_won, player_died)
    state = state.replace(cumulative_reward=state.cumulative_reward + reward)

    obs = get_observation(state)
    return state, obs, reward, done


def get_valid_actions(state: EnvState) -> jax.Array:
    """
    Return mask of valid actions.

    Args:
        state: Current environment state

    Returns:
        Boolean array of shape (NUM_ACTIONS,) where True = valid
    """
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)

    # Movement actions (0-3): check bounds and obstacles
    mask = mask.at[ACTION_MOVE_UP].set(_is_move_valid(state, ACTION_MOVE_UP))
    mask = mask.at[ACTION_MOVE_DOWN].set(_is_move_valid(state, ACTION_MOVE_DOWN))
    mask = mask.at[ACTION_MOVE_LEFT].set(_is_move_valid(state, ACTION_MOVE_LEFT))
    mask = mask.at[ACTION_MOVE_RIGHT].set(_is_move_valid(state, ACTION_MOVE_RIGHT))

    # Siphon (4): requires data_siphons > 0
    mask = mask.at[ACTION_SIPHON].set(state.player.data_siphons > 0)

    # Programs (5-27): requires ownership, resources, and applicability
    for prog_idx in range(NUM_PROGRAMS):
        action_idx = ACTION_PROGRAM_START + prog_idx
        mask = mask.at[action_idx].set(_is_program_valid(state, prog_idx))

    return mask


# =============================================================================
# Action Execution
# =============================================================================


def _execute_action(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute an action, return (new_state, turn_ends)."""

    # Branch on action type
    is_move = action < 4
    is_siphon = action == 4
    is_program = action >= 5

    # Execute move
    state_after_move, turn_ends_move = jax.lax.cond(
        is_move,
        lambda: _execute_move(state, action, key),
        lambda: (state, jnp.bool_(False)),
    )

    # Execute siphon
    state_after_siphon, turn_ends_siphon = jax.lax.cond(
        is_siphon,
        lambda: _execute_siphon(state_after_move, key),
        lambda: (state_after_move, jnp.bool_(False)),
    )

    # Execute program
    state_after_program, turn_ends_program = jax.lax.cond(
        is_program,
        lambda: _execute_program(state_after_siphon, action - ACTION_PROGRAM_START, key),
        lambda: (state_after_siphon, jnp.bool_(False)),
    )

    # Determine if turn ends
    turn_ends = jax.lax.cond(
        is_move,
        lambda: turn_ends_move,
        lambda: jax.lax.cond(
            is_siphon,
            lambda: turn_ends_siphon,
            lambda: turn_ends_program,
        ),
    )

    return state_after_program, turn_ends


def _execute_move(
    state: EnvState, direction: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a movement action.

    Movement can result in:
    1. Attack if enemy in line of sight
    2. Move to target cell if empty
    """
    offset = DIRECTION_OFFSETS[direction]
    target_row = state.player.row + offset[0]
    target_col = state.player.col + offset[1]

    # Check for enemy in line of sight
    has_enemy, enemy_idx = _find_enemy_in_los(state, direction)

    # If enemy found, attack it
    state = jax.lax.cond(
        has_enemy,
        lambda s: _attack_enemy(s, enemy_idx),
        lambda s: _move_player(s, target_row, target_col),
        state,
    )

    # Increment turn counter
    state = state.replace(turn=state.turn + 1)

    return state, jnp.bool_(True)


def _execute_siphon(state: EnvState, key: jax.Array) -> tuple[EnvState, jnp.bool_]:
    """Execute siphon action on best adjacent block.

    Siphon priority: program > high-point data > low-point data
    Effects:
    - Mark block as siphoned
    - Mark siphon center (player position)
    - Add score for data blocks
    - Add program for program blocks
    - Spawn transmissions based on block's spawn count
    - Delay next scheduled task by 5 turns
    """
    from .jax_state import (
        BLOCK_DATA, BLOCK_PROGRAM, BLOCK_QUESTION, MAX_TRANSMISSIONS,
        SIPHON_DELAY_TURNS, ENEMY_VIRUS
    )

    # Find best adjacent block
    # Adjacent cells: up, down, left, right
    player_row = state.player.row
    player_col = state.player.col

    # Check each adjacent cell
    def check_cell(row, col):
        in_bounds = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)
        block_type = jax.lax.cond(
            in_bounds,
            lambda: state.grid_block_type[row, col],
            lambda: jnp.int32(0),
        )
        is_siphoned = jax.lax.cond(
            in_bounds,
            lambda: state.grid_block_siphoned[row, col],
            lambda: jnp.bool_(True),
        )
        has_block = (block_type != BLOCK_EMPTY) & ~is_siphoned
        points = jax.lax.cond(
            in_bounds & has_block,
            lambda: state.grid_block_points[row, col],
            lambda: jnp.int32(0),
        )
        is_program = block_type == BLOCK_PROGRAM
        # Score: programs get 100, data gets points
        score = jax.lax.cond(is_program, lambda: jnp.int32(100), lambda: points)
        return has_block, score, row, col

    # Check all 4 directions
    up_valid, up_score, up_row, up_col = check_cell(player_row + 1, player_col)
    down_valid, down_score, down_row, down_col = check_cell(player_row - 1, player_col)
    left_valid, left_score, left_row, left_col = check_cell(player_row, player_col - 1)
    right_valid, right_score, right_row, right_col = check_cell(player_row, player_col + 1)

    # Find best (highest score)
    best_score = jnp.int32(0)
    best_row = jnp.int32(-1)
    best_col = jnp.int32(-1)
    has_target = jnp.bool_(False)

    # Check up
    update_up = up_valid & (up_score > best_score)
    best_score = jax.lax.cond(update_up, lambda: up_score, lambda: best_score)
    best_row = jax.lax.cond(update_up, lambda: up_row, lambda: best_row)
    best_col = jax.lax.cond(update_up, lambda: up_col, lambda: best_col)
    has_target = has_target | up_valid

    # Check down
    update_down = down_valid & (down_score > best_score)
    best_score = jax.lax.cond(update_down, lambda: down_score, lambda: best_score)
    best_row = jax.lax.cond(update_down, lambda: down_row, lambda: best_row)
    best_col = jax.lax.cond(update_down, lambda: down_col, lambda: best_col)
    has_target = has_target | down_valid

    # Check left
    update_left = left_valid & (left_score > best_score)
    best_score = jax.lax.cond(update_left, lambda: left_score, lambda: best_score)
    best_row = jax.lax.cond(update_left, lambda: left_row, lambda: best_row)
    best_col = jax.lax.cond(update_left, lambda: left_col, lambda: best_col)
    has_target = has_target | left_valid

    # Check right
    update_right = right_valid & (right_score > best_score)
    best_score = jax.lax.cond(update_right, lambda: right_score, lambda: best_score)
    best_row = jax.lax.cond(update_right, lambda: right_row, lambda: best_row)
    best_col = jax.lax.cond(update_right, lambda: right_col, lambda: best_col)
    has_target = has_target | right_valid

    # Apply siphon effects
    def apply_siphon(state):
        block_type = state.grid_block_type[best_row, best_col]
        block_points = state.grid_block_points[best_row, best_col]
        block_program = state.grid_block_program[best_row, best_col]
        spawn_count = state.grid_block_spawn_count[best_row, best_col]

        # Mark block as siphoned
        new_siphoned = state.grid_block_siphoned.at[best_row, best_col].set(True)

        # Mark siphon center (player position)
        new_siphon_center = state.grid_siphon_center.at[player_row, player_col].set(True)

        # Add score for data blocks
        new_score = jax.lax.cond(
            block_type == BLOCK_DATA,
            lambda: state.player.score + block_points,
            lambda: state.player.score,
        )

        # Add program for program blocks
        new_owned = jax.lax.cond(
            block_type == BLOCK_PROGRAM,
            lambda: state.owned_programs.at[block_program].set(True),
            lambda: state.owned_programs,
        )

        # Decrement data siphons
        new_siphons = state.player.data_siphons - 1

        player = state.player.replace(
            score=new_score,
            data_siphons=new_siphons,
        )

        # Delay scheduled tasks
        new_next_task = state.next_scheduled_task_turn + SIPHON_DELAY_TURNS

        state = state.replace(
            player=player,
            grid_block_siphoned=new_siphoned,
            grid_siphon_center=new_siphon_center,
            owned_programs=new_owned,
            next_scheduled_task_turn=new_next_task,
        )

        # Spawn transmissions
        key_for_trans = state.rng_key

        def spawn_transmission(carry, i):
            s, key = carry
            should_spawn = i < spawn_count

            # Find empty cell for transmission (use player position + offset)
            # Simplified: spawn at block position
            trans_row = best_row
            trans_col = best_col

            # Find empty slot
            slot = jnp.argmin(s.trans_mask)
            has_space = ~s.trans_mask.all()

            # Random enemy type (simplified: always virus)
            key, subkey = jax.random.split(key)
            enemy_type = jax.random.randint(subkey, (), 0, 4)

            trans_data = jnp.array([
                trans_row, trans_col, 3, enemy_type, 1, 0  # 3 turns, spawned_from_siphon=True
            ], dtype=jnp.int32)

            new_trans = jax.lax.cond(
                should_spawn & has_space,
                lambda: s.transmissions.at[slot].set(trans_data),
                lambda: s.transmissions,
            )
            new_mask = jax.lax.cond(
                should_spawn & has_space,
                lambda: s.trans_mask.at[slot].set(True),
                lambda: s.trans_mask,
            )

            return (s.replace(transmissions=new_trans, trans_mask=new_mask, rng_key=key), key), None

        (state, _), _ = jax.lax.scan(
            spawn_transmission,
            (state, key_for_trans),
            jnp.arange(9),  # Max spawn count is 9
        )

        return state

    # Only apply if we found a target, otherwise just decrement siphons
    state = jax.lax.cond(
        has_target,
        apply_siphon,
        lambda s: s.replace(
            player=s.player.replace(data_siphons=s.player.data_siphons - 1)
        ),
        state,
    )

    # Increment turn
    state = state.replace(turn=state.turn + 1)

    return state, jnp.bool_(True)


def _execute_program(
    state: EnvState, prog_idx: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a program.

    Programs (0-22):
    0: PUSH - push enemies away
    1: PULL - pull enemies closer
    2: CRASH - damage all enemies
    3: WARP - teleport to random cell
    4: POLY - transform random enemy
    5: WAIT - end turn (free)
    6: DEBUG - stun adjacent enemies
    7: ROW - damage enemies in row
    8: COL - damage enemies in column
    9: UNDO - restore previous state
    10: STEP - enemies skip movement next turn
    11: SIPH+ - add 1 data siphon
    12: EXCH - swap credits/energy
    13: SHOW - reveal cryptogs
    14: RESET - restore HP to 3
    15: CALM - disable scheduled tasks
    16: D_BOM - damage based on data siphons
    17: DELAY - delay scheduled task
    18: ANTI-V - kill all viruses
    19: SCORE - convert credits to score
    20: REDUC - reduce enemy HP
    21: ATK+ - increase attack damage
    22: HACK - damage enemies on siphoned cells
    """
    from .jax_state import (
        PROGRAM_WAIT, PROGRAM_SIPH_PLUS, PROGRAM_EXCH, PROGRAM_SHOW,
        PROGRAM_RESET, PROGRAM_CALM, PROGRAM_STEP, PROGRAM_ATK_PLUS,
        PROGRAM_PUSH, PROGRAM_PULL, PROGRAM_DEBUG, PROGRAM_ROW, PROGRAM_COL,
        PROGRAM_CRASH, PROGRAM_HACK, PROGRAM_ANTI_V, PROGRAM_D_BOM,
        PLAYER_MAX_HP, ENEMY_VIRUS,
    )

    # Deduct costs
    cost = PROGRAM_COSTS[prog_idx]
    player = state.player.replace(
        credits=state.player.credits - cost[0],
        energy=state.player.energy - cost[1],
    )
    state = state.replace(player=player)

    # WAIT ends the turn
    is_wait = prog_idx == PROGRAM_WAIT
    state = jax.lax.cond(
        is_wait,
        lambda s: s.replace(turn=s.turn + 1),
        lambda s: s,
        state,
    )

    # SIPH+ - add 1 data siphon
    state = jax.lax.cond(
        prog_idx == PROGRAM_SIPH_PLUS,
        lambda s: s.replace(
            player=s.player.replace(data_siphons=s.player.data_siphons + 1)
        ),
        lambda s: s,
        state,
    )

    # RESET - restore HP to max
    state = jax.lax.cond(
        prog_idx == PROGRAM_RESET,
        lambda s: s.replace(
            player=s.player.replace(hp=jnp.int32(PLAYER_MAX_HP))
        ),
        lambda s: s,
        state,
    )

    # STEP - set step_active flag (enemies skip movement)
    state = jax.lax.cond(
        prog_idx == PROGRAM_STEP,
        lambda s: s.replace(step_active=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # SHOW - reveal cryptogs
    state = jax.lax.cond(
        prog_idx == PROGRAM_SHOW,
        lambda s: s.replace(show_activated=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # CALM - disable scheduled tasks
    state = jax.lax.cond(
        prog_idx == PROGRAM_CALM,
        lambda s: s.replace(scheduled_tasks_disabled=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # ATK+ - increase attack damage (max 2 uses per stage)
    can_use_atkplus = (prog_idx == PROGRAM_ATK_PLUS) & (state.atk_plus_uses_this_stage < 2)
    state = jax.lax.cond(
        can_use_atkplus,
        lambda s: s.replace(
            player=s.player.replace(
                attack_damage=jnp.minimum(s.player.attack_damage + 1, 3)
            ),
            atk_plus_uses_this_stage=s.atk_plus_uses_this_stage + 1,
        ),
        lambda s: s,
        state,
    )

    # EXCH - swap credits and energy
    state = jax.lax.cond(
        prog_idx == PROGRAM_EXCH,
        lambda s: s.replace(
            player=s.player.replace(
                credits=s.player.energy,
                energy=s.player.credits,
            )
        ),
        lambda s: s,
        state,
    )

    # DEBUG - stun adjacent enemies
    def apply_debug(s):
        def stun_if_adjacent(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row = enemy[1]
            col = enemy[2]
            row_dist = jnp.abs(row - state.player.row)
            col_dist = jnp.abs(col - state.player.col)
            is_adjacent = (row_dist + col_dist) == 1

            should_stun = is_active & is_adjacent
            new_enemies = jax.lax.cond(
                should_stun,
                lambda: state.enemies.at[idx, 5].set(1),  # Set stunned
                lambda: state.enemies,
            )
            return state.replace(enemies=new_enemies), None

        state_out, _ = jax.lax.scan(stun_if_adjacent, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_DEBUG,
        apply_debug,
        lambda s: s,
        state,
    )

    # ROW - damage enemies in player's row
    def apply_row(s):
        def damage_in_row(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row = enemy[1]
            in_row = row == state.player.row

            should_damage = is_active & in_row
            new_hp = jnp.maximum(enemy[3] - 1, 0)
            new_enemies = jax.lax.cond(
                should_damage,
                lambda: state.enemies.at[idx, 3].set(new_hp),
                lambda: state.enemies,
            )
            # Remove if dead
            new_mask = jax.lax.cond(
                should_damage & (new_hp <= 0),
                lambda: state.enemy_mask.at[idx].set(False),
                lambda: state.enemy_mask,
            )
            return state.replace(enemies=new_enemies, enemy_mask=new_mask), None

        state_out, _ = jax.lax.scan(damage_in_row, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_ROW,
        apply_row,
        lambda s: s,
        state,
    )

    # COL - damage enemies in player's column
    def apply_col(s):
        def damage_in_col(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            col = enemy[2]
            in_col = col == state.player.col

            should_damage = is_active & in_col
            new_hp = jnp.maximum(enemy[3] - 1, 0)
            new_enemies = jax.lax.cond(
                should_damage,
                lambda: state.enemies.at[idx, 3].set(new_hp),
                lambda: state.enemies,
            )
            new_mask = jax.lax.cond(
                should_damage & (new_hp <= 0),
                lambda: state.enemy_mask.at[idx].set(False),
                lambda: state.enemy_mask,
            )
            return state.replace(enemies=new_enemies, enemy_mask=new_mask), None

        state_out, _ = jax.lax.scan(damage_in_col, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_COL,
        apply_col,
        lambda s: s,
        state,
    )

    # CRASH - damage ALL enemies (1 damage)
    def apply_crash(s):
        def damage_enemy(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            new_hp = jnp.maximum(enemy[3] - 1, 0)
            new_enemies = jax.lax.cond(
                is_active,
                lambda: state.enemies.at[idx, 3].set(new_hp),
                lambda: state.enemies,
            )
            new_mask = jax.lax.cond(
                is_active & (new_hp <= 0),
                lambda: state.enemy_mask.at[idx].set(False),
                lambda: state.enemy_mask,
            )
            return state.replace(enemies=new_enemies, enemy_mask=new_mask), None

        state_out, _ = jax.lax.scan(damage_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_CRASH,
        apply_crash,
        lambda s: s,
        state,
    )

    # ANTI-V - kill all viruses
    def apply_antiv(s):
        def kill_virus(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            is_virus = enemy[0] == ENEMY_VIRUS

            should_kill = is_active & is_virus
            new_enemies = jax.lax.cond(
                should_kill,
                lambda: state.enemies.at[idx, 3].set(0),
                lambda: state.enemies,
            )
            new_mask = jax.lax.cond(
                should_kill,
                lambda: state.enemy_mask.at[idx].set(False),
                lambda: state.enemy_mask,
            )
            return state.replace(enemies=new_enemies, enemy_mask=new_mask), None

        state_out, _ = jax.lax.scan(kill_virus, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_ANTI_V,
        apply_antiv,
        lambda s: s,
        state,
    )

    # HACK - damage enemies on siphoned cells
    def apply_hack(s):
        def damage_on_siphoned(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row = enemy[1]
            col = enemy[2]

            # Check if enemy is on a siphoned block
            on_siphoned = state.grid_block_siphoned[row, col]

            should_damage = is_active & on_siphoned
            # HACK does 2 damage
            new_hp = jnp.maximum(enemy[3] - 2, 0)
            new_enemies = jax.lax.cond(
                should_damage,
                lambda: state.enemies.at[idx, 3].set(new_hp),
                lambda: state.enemies,
            )
            new_mask = jax.lax.cond(
                should_damage & (new_hp <= 0),
                lambda: state.enemy_mask.at[idx].set(False),
                lambda: state.enemy_mask,
            )
            return state.replace(enemies=new_enemies, enemy_mask=new_mask), None

        state_out, _ = jax.lax.scan(damage_on_siphoned, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_HACK,
        apply_hack,
        lambda s: s,
        state,
    )

    # PUSH - push enemies away from player
    def apply_push(s):
        def push_enemy(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row = enemy[1]
            col = enemy[2]

            # Push direction (away from player)
            row_delta = jnp.sign(row - state.player.row)
            col_delta = jnp.sign(col - state.player.col)

            # New position (push by 1 cell)
            new_row = jnp.clip(row + row_delta, 0, GRID_SIZE - 1)
            new_col = jnp.clip(col + col_delta, 0, GRID_SIZE - 1)

            # Update if active
            new_enemies = jax.lax.cond(
                is_active,
                lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
                lambda: state.enemies,
            )
            return state.replace(enemies=new_enemies), None

        state_out, _ = jax.lax.scan(push_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_PUSH,
        apply_push,
        lambda s: s,
        state,
    )

    # PULL - pull enemies toward player
    def apply_pull(s):
        def pull_enemy(carry, idx):
            state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row = enemy[1]
            col = enemy[2]

            # Pull direction (toward player)
            row_delta = jnp.sign(state.player.row - row)
            col_delta = jnp.sign(state.player.col - col)

            # New position (pull by 1 cell)
            new_row = jnp.clip(row + row_delta, 0, GRID_SIZE - 1)
            new_col = jnp.clip(col + col_delta, 0, GRID_SIZE - 1)

            # Update if active
            new_enemies = jax.lax.cond(
                is_active,
                lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
                lambda: state.enemies,
            )
            return state.replace(enemies=new_enemies), None

        state_out, _ = jax.lax.scan(pull_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
        return state_out

    state = jax.lax.cond(
        prog_idx == PROGRAM_PULL,
        apply_pull,
        lambda s: s,
        state,
    )

    return state, is_wait


# =============================================================================
# Movement Helpers
# =============================================================================


def _is_move_valid(state: EnvState, direction: jnp.int32) -> jnp.bool_:
    """Check if movement in direction is valid.

    Valid if:
    - Target cell is in bounds AND
    - (Target cell has no unsiphoned block OR there's an enemy in LOS)
    """
    offset = DIRECTION_OFFSETS[direction]
    target_row = state.player.row + offset[0]
    target_col = state.player.col + offset[1]

    # Check bounds
    in_bounds = (
        (target_row >= 0) & (target_row < GRID_SIZE) &
        (target_col >= 0) & (target_col < GRID_SIZE)
    )

    # Check for blocking block at target
    has_blocking_block = jax.lax.cond(
        in_bounds,
        lambda: (state.grid_block_type[target_row, target_col] != BLOCK_EMPTY) &
                (~state.grid_block_siphoned[target_row, target_col]),
        lambda: jnp.bool_(False),
    )

    # Check for enemy in line of sight (allows "attack move")
    has_los_enemy, _ = _find_enemy_in_los(state, direction)

    return in_bounds & (~has_blocking_block | has_los_enemy)


def _find_enemy_in_los(
    state: EnvState, direction: jnp.int32
) -> tuple[jnp.bool_, jnp.int32]:
    """Find first enemy or transmission in line of sight.

    Returns (found, enemy_idx). If found is False, enemy_idx is invalid.
    """
    offset = DIRECTION_OFFSETS[direction]

    # Search along direction for enemy
    def scan_cell(carry, step):
        found, idx, state, offset = carry
        row = state.player.row + offset[0] * (step + 1)
        col = state.player.col + offset[1] * (step + 1)

        # Check bounds
        in_bounds = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)

        # Check for blocking block
        has_block = jax.lax.cond(
            in_bounds,
            lambda: (state.grid_block_type[row, col] != BLOCK_EMPTY) &
                    (~state.grid_block_siphoned[row, col]),
            lambda: jnp.bool_(False),
        )

        # Check for enemy at this cell
        def check_enemy(enemy_found_idx):
            enemy_found, enemy_idx = enemy_found_idx
            for i in range(state.enemy_mask.shape[0]):
                is_at_cell = (
                    state.enemy_mask[i] &
                    (state.enemies[i, 1] == row) &
                    (state.enemies[i, 2] == col)
                )
                enemy_found = enemy_found | is_at_cell
                enemy_idx = jax.lax.cond(
                    is_at_cell & (enemy_idx < 0),
                    lambda: jnp.int32(i),
                    lambda: enemy_idx,
                )
            return enemy_found, enemy_idx

        # Use fori_loop for enemy check
        def check_enemies_at_cell(carry, i):
            enemy_found, enemy_idx, row, col, state = carry
            is_at_cell = (
                state.enemy_mask[i] &
                (state.enemies[i, 1] == row) &
                (state.enemies[i, 2] == col)
            )
            new_found = enemy_found | is_at_cell
            new_idx = jax.lax.cond(
                is_at_cell & ~enemy_found,
                lambda: jnp.int32(i),
                lambda: enemy_idx,
            )
            return (new_found, new_idx, row, col, state), None

        enemy_result = jax.lax.cond(
            in_bounds & ~found & ~has_block,
            lambda: jax.lax.scan(
                check_enemies_at_cell,
                (jnp.bool_(False), jnp.int32(-1), row, col, state),
                jnp.arange(state.enemy_mask.shape[0]),
            )[0][:2],
            lambda: (jnp.bool_(False), jnp.int32(-1)),
        )

        # Update found status - stop if blocked or found
        new_found = found | enemy_result[0]
        new_idx = jax.lax.cond(
            enemy_result[0] & ~found,
            lambda: enemy_result[1],
            lambda: idx,
        )

        # Stop searching if we hit a block
        stop = ~in_bounds | has_block | new_found

        return (new_found, new_idx, state, offset), stop

    # Scan up to GRID_SIZE cells
    (found, idx, _, _), _ = jax.lax.scan(
        scan_cell,
        (jnp.bool_(False), jnp.int32(-1), state, offset),
        jnp.arange(GRID_SIZE),
    )

    return found, idx


def _move_player(state: EnvState, target_row: jnp.int32, target_col: jnp.int32) -> EnvState:
    """Move player to target cell and collect pickups."""
    # Check bounds
    in_bounds = (
        (target_row >= 0) & (target_row < GRID_SIZE) &
        (target_col >= 0) & (target_col < GRID_SIZE)
    )

    new_row = jax.lax.cond(in_bounds, lambda: target_row, lambda: state.player.row)
    new_col = jax.lax.cond(in_bounds, lambda: target_col, lambda: state.player.col)

    # Collect data siphon if present
    collected_siphon = jax.lax.cond(
        in_bounds,
        lambda: state.grid_data_siphon[new_row, new_col],
        lambda: jnp.bool_(False),
    )

    new_siphons = jax.lax.cond(
        collected_siphon,
        lambda: state.player.data_siphons + 1,
        lambda: state.player.data_siphons,
    )

    # Clear the data siphon from grid
    new_grid_data_siphon = jax.lax.cond(
        collected_siphon,
        lambda: state.grid_data_siphon.at[new_row, new_col].set(False),
        lambda: state.grid_data_siphon,
    )

    player = state.player.replace(
        row=new_row,
        col=new_col,
        data_siphons=new_siphons,
    )

    return state.replace(player=player, grid_data_siphon=new_grid_data_siphon)


def _attack_enemy(state: EnvState, enemy_idx: jnp.int32) -> EnvState:
    """Attack an enemy, dealing player's attack damage."""
    enemy = state.enemies[enemy_idx]
    new_hp = enemy[3] - state.player.attack_damage

    # Update enemy HP
    new_enemies = state.enemies.at[enemy_idx, 3].set(jnp.maximum(new_hp, 0))

    # Remove enemy if dead
    enemy_killed = new_hp <= 0
    new_mask = jax.lax.cond(
        enemy_killed,
        lambda: state.enemy_mask.at[enemy_idx].set(False),
        lambda: state.enemy_mask,
    )

    # TODO: Add score for kills

    return state.replace(enemies=new_enemies, enemy_mask=new_mask)


# =============================================================================
# Program Validation
# =============================================================================


def _is_program_valid(state: EnvState, prog_idx: jnp.int32) -> jnp.bool_:
    """Check if a program can be used."""
    # Must own the program
    owns_program = state.owned_programs[prog_idx]

    # Must have enough resources
    cost = PROGRAM_COSTS[prog_idx]
    has_credits = state.player.credits >= cost[0]
    has_energy = state.player.energy >= cost[1]

    # TODO: Add program-specific applicability checks

    return owns_program & has_credits & has_energy


# =============================================================================
# Enemy Turn Processing
# =============================================================================


def _process_enemy_turn(state: EnvState) -> EnvState:
    """Process enemy turn after player action ends their turn."""
    # 1. Tick transmissions and spawn enemies
    state = _tick_transmissions(state)

    # 2. Move enemies (unless STEP is active)
    state = jax.lax.cond(
        ~state.step_active,
        lambda s: _move_enemies(s),
        lambda s: s,
        state,
    )

    # 3. Enemy attacks
    state = _enemy_attacks(state)

    # 4. Check scheduled tasks
    state = _check_scheduled_tasks(state)

    # 5. Reset status effects
    state = _reset_status_effects(state)

    # Clear STEP effect after enemy turn
    state = state.replace(step_active=jnp.bool_(False))

    return state


def _tick_transmissions(state: EnvState) -> EnvState:
    """Decrement transmission timers and spawn enemies."""
    from .jax_state import MAX_ENEMIES, ENEMY_MAX_HP

    def process_transmission(carry, idx):
        state = carry
        is_active = state.trans_mask[idx]
        trans = state.transmissions[idx]
        row = trans[0]
        col = trans[1]
        turns_remaining = trans[2]
        enemy_type = trans[3]
        spawned_from_siphon = trans[4]
        is_from_scheduled_task = trans[5]

        # Decrement turns
        new_turns = turns_remaining - 1

        # If timer hits 0, spawn enemy
        should_spawn = is_active & (new_turns <= 0)

        # Update transmission turns
        new_transmissions = jax.lax.cond(
            is_active & (new_turns > 0),
            lambda: state.transmissions.at[idx, 2].set(new_turns),
            lambda: state.transmissions,
        )

        # Remove transmission if spawned
        new_trans_mask = jax.lax.cond(
            should_spawn,
            lambda: state.trans_mask.at[idx].set(False),
            lambda: state.trans_mask,
        )

        # Find slot for new enemy
        enemy_slot = jnp.argmin(state.enemy_mask)
        has_enemy_space = ~state.enemy_mask.all()

        # Create enemy data
        enemy_hp = ENEMY_MAX_HP[enemy_type]
        enemy_data = jnp.array([
            enemy_type, row, col, enemy_hp, 0,
            0, spawned_from_siphon, is_from_scheduled_task
        ], dtype=jnp.int32)

        # Add enemy if spawning and space available
        new_enemies = jax.lax.cond(
            should_spawn & has_enemy_space,
            lambda: state.enemies.at[enemy_slot].set(enemy_data),
            lambda: state.enemies,
        )
        new_enemy_mask = jax.lax.cond(
            should_spawn & has_enemy_space,
            lambda: state.enemy_mask.at[enemy_slot].set(True),
            lambda: state.enemy_mask,
        )

        new_state = state.replace(
            transmissions=new_transmissions,
            trans_mask=new_trans_mask,
            enemies=new_enemies,
            enemy_mask=new_enemy_mask,
        )

        return new_state, None

    state, _ = jax.lax.scan(
        process_transmission,
        state,
        jnp.arange(state.trans_mask.shape[0]),
    )

    return state


def _move_enemies(state: EnvState) -> EnvState:
    """Move all non-stunned enemies toward player.

    Simple pathfinding: move directly toward player.
    Virus moves 2 cells, others move 1 cell.
    Glitch can move through blocks.
    """
    from .jax_state import ENEMY_VIRUS, ENEMY_GLITCH, ENEMY_SPEED

    def move_single_enemy(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        enemy_type = enemy[0]
        row = enemy[1]
        col = enemy[2]
        disabled_turns = enemy[4]
        is_stunned = enemy[5] > 0

        # Check if enemy can move
        can_move = is_active & ~is_stunned & (disabled_turns <= 0)

        # Calculate direction toward player
        row_diff = state.player.row - row
        col_diff = state.player.col - col

        # Get movement speed for this enemy type
        speed = ENEMY_SPEED[enemy_type]

        # Move function - moves one step toward player
        def do_move(state_row_col):
            s, r, c = state_row_col
            # Prefer row movement if row diff is larger, else col
            row_delta = jnp.sign(s.player.row - r)
            col_delta = jnp.sign(s.player.col - c)

            # Try to move in primary direction
            # For simplicity, alternate between row and col movement
            abs_row_diff = jnp.abs(s.player.row - r)
            abs_col_diff = jnp.abs(s.player.col - c)

            # Move in direction of larger distance
            new_row = jax.lax.cond(
                abs_row_diff >= abs_col_diff,
                lambda: r + row_delta,
                lambda: r,
            )
            new_col = jax.lax.cond(
                abs_row_diff < abs_col_diff,
                lambda: c + col_delta,
                lambda: c,
            )

            # Bounds check
            new_row = jnp.clip(new_row, 0, GRID_SIZE - 1)
            new_col = jnp.clip(new_col, 0, GRID_SIZE - 1)

            # Check for blocking block (unless glitch)
            is_glitch = enemy_type == ENEMY_GLITCH
            has_block = (s.grid_block_type[new_row, new_col] != BLOCK_EMPTY) & \
                       (~s.grid_block_siphoned[new_row, new_col])
            blocked = has_block & ~is_glitch

            # If blocked, try alternate direction
            alt_row = jax.lax.cond(
                blocked & (abs_row_diff < abs_col_diff),
                lambda: r + row_delta,
                lambda: new_row,
            )
            alt_col = jax.lax.cond(
                blocked & (abs_row_diff >= abs_col_diff),
                lambda: c + col_delta,
                lambda: new_col,
            )

            # Use alternate if not blocked
            final_row = jax.lax.cond(
                blocked,
                lambda: jnp.clip(alt_row, 0, GRID_SIZE - 1),
                lambda: new_row,
            )
            final_col = jax.lax.cond(
                blocked,
                lambda: jnp.clip(alt_col, 0, GRID_SIZE - 1),
                lambda: new_col,
            )

            return final_row, final_col

        # Move up to 'speed' times
        new_row, new_col = row, col
        new_row, new_col = jax.lax.cond(
            can_move,
            lambda: do_move((state, new_row, new_col)),
            lambda: (new_row, new_col),
        )

        # Virus moves twice
        new_row, new_col = jax.lax.cond(
            can_move & (enemy_type == ENEMY_VIRUS),
            lambda: do_move((state, new_row, new_col)),
            lambda: (new_row, new_col),
        )

        # Update enemy position
        new_enemies = jax.lax.cond(
            can_move,
            lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
            lambda: state.enemies,
        )

        return state.replace(enemies=new_enemies), None

    state, _ = jax.lax.scan(
        move_single_enemy,
        state,
        jnp.arange(state.enemy_mask.shape[0]),
    )

    return state


def _enemy_attacks(state: EnvState) -> EnvState:
    """Adjacent enemies attack player."""
    # Count adjacent non-stunned enemies
    def count_adjacent_attackers(carry, idx):
        count, state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        is_stunned = enemy[5] > 0

        # Check if adjacent (Manhattan distance = 1)
        row_dist = jnp.abs(row - state.player.row)
        col_dist = jnp.abs(col - state.player.col)
        is_adjacent = (row_dist + col_dist) == 1

        should_attack = is_active & is_adjacent & ~is_stunned
        new_count = count + jax.lax.cond(should_attack, lambda: 1, lambda: 0)

        return (new_count, state), None

    (attack_count, _), _ = jax.lax.scan(
        count_adjacent_attackers,
        (0, state),
        jnp.arange(state.enemy_mask.shape[0]),
    )

    # Reduce player HP
    new_hp = jnp.maximum(state.player.hp - attack_count, 0)
    player = state.player.replace(hp=new_hp)

    return state.replace(player=player)


def _check_scheduled_tasks(state: EnvState) -> EnvState:
    """Check and execute scheduled tasks."""
    # TODO: Implement scheduled task spawning
    return state


def _reset_status_effects(state: EnvState) -> EnvState:
    """Reset enemy stun flags and decrement disable counters."""
    # Clear stun flags
    new_enemies = state.enemies.at[:, 5].set(0)

    # Decrement disable counters (column 4)
    disable_counters = state.enemies[:, 4]
    new_disable = jnp.maximum(disable_counters - 1, 0)
    new_enemies = new_enemies.at[:, 4].set(new_disable)

    return state.replace(enemies=new_enemies)


# =============================================================================
# State Management
# =============================================================================


def _save_previous_state(state: EnvState) -> EnvState:
    """Save current state for UNDO functionality."""
    return state.replace(
        previous_state_valid=jnp.bool_(True),
        previous_player=state.player,
        previous_enemies=state.enemies,
        previous_enemy_mask=state.enemy_mask,
        previous_transmissions=state.transmissions,
        previous_trans_mask=state.trans_mask,
        previous_turn=state.turn,
        previous_grid_block_siphoned=state.grid_block_siphoned,
        previous_grid_siphon_center=state.grid_siphon_center,
        prev_score=state.player.score,
        prev_hp=state.player.hp,
        prev_credits=state.player.credits,
        prev_energy=state.player.energy,
    )


def _advance_stage(state: EnvState) -> EnvState:
    """Advance to next stage."""
    from .jax_state import PLAYER_MAX_HP

    new_stage = state.stage + 1

    # Reset player position and restore HP
    player = state.player.replace(
        row=jnp.int32(0),
        col=jnp.int32(0),
        hp=jnp.int32(PLAYER_MAX_HP),  # Restore HP on stage transition
    )

    # Reset stage-scoped flags
    state = state.replace(
        player=player,
        stage=new_stage,
        atk_plus_uses_this_stage=jnp.int32(0),
        step_active=jnp.bool_(False),
        show_activated=jnp.bool_(False),
    )

    # Clear grid (except exit)
    grid_shape = (GRID_SIZE, GRID_SIZE)
    state = state.replace(
        grid_block_type=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_points=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_program=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_spawn_count=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_siphoned=jnp.zeros(grid_shape, dtype=jnp.bool_),
        grid_siphon_center=jnp.zeros(grid_shape, dtype=jnp.bool_),
        grid_resources_credits=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_resources_energy=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_data_siphon=jnp.zeros(grid_shape, dtype=jnp.bool_),
    )

    # Clear enemies and transmissions
    state = state.replace(
        enemy_mask=jnp.zeros(state.enemy_mask.shape, dtype=jnp.bool_),
        trans_mask=jnp.zeros(state.trans_mask.shape, dtype=jnp.bool_),
    )

    # TODO: Generate new stage content

    return state


# =============================================================================
# Reward Calculation
# =============================================================================


def _calculate_reward(
    state: EnvState,
    stage_completed: jnp.bool_,
    game_won: jnp.bool_,
    player_died: jnp.bool_,
) -> jnp.float32:
    """Calculate reward for this transition."""
    reward = jnp.float32(0.0)

    # Stage completion reward
    from .jax_state import STAGE_COMPLETION_REWARDS
    stage_reward = jax.lax.cond(
        stage_completed & (state.stage <= 8),
        lambda: STAGE_COMPLETION_REWARDS[state.stage - 2],  # -2 because stage was incremented
        lambda: jnp.float32(0.0),
    )
    reward = reward + stage_reward

    # Score gain reward (0.5 per point)
    score_delta = state.player.score - state.prev_score
    reward = reward + score_delta * 0.5

    # HP change
    hp_delta = state.player.hp - state.prev_hp
    reward = reward + hp_delta * 1.0  # +1 per HP gained, -1 per HP lost

    # Victory bonus
    victory_bonus = jax.lax.cond(
        game_won,
        lambda: 500.0 + state.player.score * 100.0,
        lambda: jnp.float32(0.0),
    )
    reward = reward + victory_bonus

    # Death penalty
    death_penalty = jax.lax.cond(
        player_died,
        lambda: -state.cumulative_reward * 0.5,
        lambda: jnp.float32(0.0),
    )
    reward = reward + death_penalty

    return reward


# =============================================================================
# Vectorized Versions
# =============================================================================


# Note: These will be re-enabled once the basic functions work
# batched_reset = jax.vmap(reset)
# batched_step = jax.vmap(step)
# batched_get_valid_actions = jax.vmap(get_valid_actions)
