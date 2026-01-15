"""
Action Execution and State Transitions

This module handles:
1. Executing actions (pass, dribble, shoot) and producing new game states
2. Simulating opponent responses to actions
3. Running the decision loop: state → evaluate → decide → execute → repeat

Key design: All players have EQUAL abilities. The engine finds optimal decisions
based purely on POSITION and SPACE, not individual talent.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import math
from copy import deepcopy

from .pitch_geometry import Position, Velocity, PitchGeometry, HALF_LENGTH, HALF_WIDTH
from .elimination import Player
from .state_scoring import (
    GameState,
    GameStateEvaluator,
    ActionOption,
    ActionType,
)


# Constants for equal-ability players
PLAYER_SPEED = 8.0          # m/s
PASS_SPEED = 15.0           # m/s
DRIBBLE_SPEED = 6.0         # m/s (slower than running)
REACTION_TIME = 0.25        # seconds
TIME_STEP = 0.5             # seconds per simulation step


class ActionResult(Enum):
    """Outcome of an executed action."""
    SUCCESS = "success"
    INTERCEPTED = "intercepted"
    OUT_OF_BOUNDS = "out_of_bounds"
    GOAL = "goal"
    SAVED = "saved"


@dataclass
class ExecutionResult:
    """Result of executing an action."""
    action: ActionOption
    result: ActionResult
    new_state: GameState
    message: str = ""


@dataclass
class DecisionStep:
    """Record of one decision in the sequence."""
    state_before: GameState
    action_chosen: ActionOption
    result: ExecutionResult
    state_after: GameState


class ActionExecutor:
    """
    Executes actions and produces new game states.

    This is deterministic for now - actions succeed or fail based on
    clear rules, not randomness. This makes the system predictable
    and debuggable.
    """

    def __init__(self, geometry: Optional[PitchGeometry] = None):
        self.geometry = geometry or PitchGeometry()

    def execute(
        self,
        state: GameState,
        action: ActionOption
    ) -> ExecutionResult:
        """
        Execute an action and return the resulting state.

        Args:
            state: Current game state
            action: Action to execute

        Returns:
            ExecutionResult with new state and outcome
        """
        if action.action_type == ActionType.SHOT:
            return self._execute_shot(state, action)
        elif action.action_type in (
            ActionType.PASS_FORWARD,
            ActionType.PASS_LATERAL,
            ActionType.PASS_BACKWARD
        ):
            return self._execute_pass(state, action)
        elif action.action_type in (ActionType.DRIBBLE, ActionType.CARRY):
            return self._execute_dribble(state, action)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    def _execute_pass(
        self,
        state: GameState,
        action: ActionOption
    ) -> ExecutionResult:
        """Execute a pass to target position."""
        target = action.target

        # Check for interception
        interceptor = self._check_interception(
            state.ball_position,
            target,
            state.defenders
        )

        if interceptor:
            # Pass intercepted - defender gets ball
            new_state = self._create_new_state(
                state,
                ball_position=interceptor.position,
                ball_carrier=interceptor,
                possession_change=True
            )
            return ExecutionResult(
                action=action,
                result=ActionResult.INTERCEPTED,
                new_state=new_state,
                message=f"Pass intercepted by {interceptor.id}"
            )

        # Find receiving player (if any teammate at target)
        receiver = self._find_player_at(target, state.attackers)

        # Pass successful
        new_state = self._create_new_state(
            state,
            ball_position=target,
            ball_carrier=receiver,
            possession_change=False
        )

        return ExecutionResult(
            action=action,
            result=ActionResult.SUCCESS,
            new_state=new_state,
            message=f"Pass to {receiver.id if receiver else 'space'} successful"
        )

    def _execute_dribble(
        self,
        state: GameState,
        action: ActionOption
    ) -> ExecutionResult:
        """Execute a dribble to target position."""
        target = action.target

        # Check if tackled during dribble
        tackler = self._check_tackle(
            state.ball_position,
            target,
            state.defenders
        )

        if tackler:
            # Tackled - defender gets ball
            new_state = self._create_new_state(
                state,
                ball_position=tackler.position,
                ball_carrier=tackler,
                possession_change=True
            )
            return ExecutionResult(
                action=action,
                result=ActionResult.INTERCEPTED,
                new_state=new_state,
                message=f"Tackled by {tackler.id}"
            )

        # Dribble successful
        new_carrier = deepcopy(state.ball_carrier)
        if new_carrier:
            new_carrier.position = target

        new_state = self._create_new_state(
            state,
            ball_position=target,
            ball_carrier=new_carrier,
            possession_change=False
        )

        return ExecutionResult(
            action=action,
            result=ActionResult.SUCCESS,
            new_state=new_state,
            message="Dribble successful"
        )

    def _execute_shot(
        self,
        state: GameState,
        action: ActionOption
    ) -> ExecutionResult:
        """Execute a shot on goal."""
        goal_pos = self.geometry.attacking_goal

        # Check if shot is blocked
        blocker = self._check_shot_block(
            state.ball_position,
            goal_pos,
            state.defenders
        )

        if blocker:
            # Shot blocked
            new_state = self._create_new_state(
                state,
                ball_position=blocker.position,
                ball_carrier=blocker,
                possession_change=True
            )
            return ExecutionResult(
                action=action,
                result=ActionResult.SAVED,
                new_state=new_state,
                message=f"Shot blocked by {blocker.id}"
            )

        # Shot on target - goal scored
        # Reset to center for kickoff
        new_state = self._create_kickoff_state(state, scored_by="attack")

        return ExecutionResult(
            action=action,
            result=ActionResult.GOAL,
            new_state=new_state,
            message="GOAL!"
        )

    def _check_interception(
        self,
        start: Position,
        end: Position,
        defenders: List[Player]
    ) -> Optional[Player]:
        """
        Check if any defender can intercept a pass.

        Interception occurs if defender can reach the pass line
        before the ball reaches that point.
        """
        pass_distance = start.distance_to(end)
        pass_time = pass_distance / PASS_SPEED

        for defender in defenders:
            # Distance from defender to pass line
            perp_dist = self.geometry.perpendicular_distance_to_line(
                defender.position, start, end
            )

            # Time for defender to reach pass line
            intercept_time = REACTION_TIME + perp_dist / PLAYER_SPEED

            # Can intercept if they reach the line before ball passes
            if intercept_time < pass_time * 0.8 and perp_dist < 3.0:
                return defender

        return None

    def _check_tackle(
        self,
        start: Position,
        end: Position,
        defenders: List[Player]
    ) -> Optional[Player]:
        """
        Check if any defender can tackle during a dribble.

        Tackle occurs if defender is close enough to challenge.
        """
        for defender in defenders:
            dist_to_start = defender.position.distance_to(start)
            dist_to_end = defender.position.distance_to(end)

            # Can tackle if within 2m of dribble path
            if dist_to_start < 2.0 or dist_to_end < 2.0:
                return defender

        return None

    def _check_shot_block(
        self,
        shot_start: Position,
        goal: Position,
        defenders: List[Player]
    ) -> Optional[Player]:
        """
        Check if any defender blocks the shot.

        Simple model: defenders within 3m of shot line can block.
        """
        for defender in defenders:
            perp_dist = self.geometry.perpendicular_distance_to_line(
                defender.position, shot_start, goal
            )

            # Close enough to block
            if perp_dist < 2.0:
                # And between shooter and goal
                dist_to_goal = defender.position.distance_to(goal)
                shooter_to_goal = shot_start.distance_to(goal)
                if dist_to_goal < shooter_to_goal:
                    return defender

        return None

    def _find_player_at(
        self,
        position: Position,
        players: List[Player],
        threshold: float = 2.0
    ) -> Optional[Player]:
        """Find a player near a position."""
        for player in players:
            if player.position.distance_to(position) < threshold:
                return player
        return None

    def _create_new_state(
        self,
        old_state: GameState,
        ball_position: Position,
        ball_carrier: Optional[Player],
        possession_change: bool
    ) -> GameState:
        """
        Create new game state after action.

        Also moves players toward their logical positions.
        """
        # Deep copy players
        new_attackers = deepcopy(old_state.attackers)
        new_defenders = deepcopy(old_state.defenders)

        # Update ball carrier position in attackers list if needed
        if ball_carrier and not possession_change:
            for i, attacker in enumerate(new_attackers):
                if attacker.id == ball_carrier.id:
                    new_attackers[i] = ball_carrier
                    break

        # Move defenders toward ball (simple reactive AI)
        new_defenders = self._move_defenders_toward_ball(
            new_defenders,
            ball_position
        )

        # Swap attacker/defender lists if possession changed
        if possession_change:
            new_attackers, new_defenders = new_defenders, new_attackers

        return GameState(
            ball_position=ball_position,
            ball_carrier=ball_carrier,
            attackers=new_attackers,
            defenders=new_defenders,
            timestamp=old_state.timestamp + TIME_STEP
        )

    def _move_defenders_toward_ball(
        self,
        defenders: List[Player],
        ball_position: Position
    ) -> List[Player]:
        """
        Move defenders toward the ball (simple reactive behavior).

        Each defender moves a fraction toward the ball.
        """
        moved = []
        for defender in defenders:
            # Vector toward ball
            dx = ball_position.x - defender.position.x
            dy = ball_position.y - defender.position.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 1.0:
                # Move at half speed toward ball
                move_dist = min(PLAYER_SPEED * TIME_STEP * 0.5, dist)
                new_x = defender.position.x + (dx / dist) * move_dist
                new_y = defender.position.y + (dy / dist) * move_dist

                new_defender = deepcopy(defender)
                new_defender.position = Position(new_x, new_y)
                moved.append(new_defender)
            else:
                moved.append(defender)

        return moved

    def _create_kickoff_state(
        self,
        old_state: GameState,
        scored_by: str
    ) -> GameState:
        """Create state for kickoff after goal."""
        # Simplified: just put ball at center
        return GameState(
            ball_position=Position(0, 0),
            ball_carrier=None,
            attackers=old_state.attackers,
            defenders=old_state.defenders,
            timestamp=old_state.timestamp + 1.0
        )


class DecisionEngine:
    """
    The main decision-making loop.

    Given a game state, this engine:
    1. Evaluates the current position
    2. Finds all possible actions
    3. Selects the best action
    4. Executes it
    5. Returns the new state

    Can also run multiple steps to simulate a sequence.
    """

    def __init__(self, geometry: Optional[PitchGeometry] = None):
        self.geometry = geometry or PitchGeometry()
        self.evaluator = GameStateEvaluator(geometry=self.geometry)
        self.executor = ActionExecutor(geometry=self.geometry)

    def decide(self, state: GameState) -> Optional[ActionOption]:
        """
        Decide the best action for the current state.

        Returns the highest expected-value action, or None if no good options.
        """
        # Evaluate state to get available actions
        evaluated = self.evaluator.evaluate(state)

        if not evaluated.available_actions:
            return None

        # Return best action (already sorted by expected value)
        return evaluated.available_actions[0]

    def step(self, state: GameState) -> DecisionStep:
        """
        Execute one decision step.

        Returns the full decision record.
        """
        # Evaluate current state
        evaluated_state = self.evaluator.evaluate(state)

        # Decide best action
        action = self.decide(evaluated_state)

        if action is None:
            # No good action - just return current state
            return DecisionStep(
                state_before=state,
                action_chosen=None,
                result=None,
                state_after=state
            )

        # Execute action
        result = self.executor.execute(state, action)

        return DecisionStep(
            state_before=state,
            action_chosen=action,
            result=result,
            state_after=result.new_state
        )

    def run(
        self,
        initial_state: GameState,
        max_steps: int = 20,
        stop_on_goal: bool = True,
        stop_on_possession_loss: bool = True
    ) -> List[DecisionStep]:
        """
        Run the decision engine for multiple steps.

        Stops when:
        - Goal is scored (if stop_on_goal)
        - Possession is lost (if stop_on_possession_loss)
        - Max steps reached
        - No valid actions available

        Returns list of all decision steps.
        """
        steps = []
        current_state = initial_state

        for i in range(max_steps):
            step = self.step(current_state)
            steps.append(step)

            # Check stopping conditions
            if step.result is None:
                break

            if stop_on_goal and step.result.result == ActionResult.GOAL:
                break

            if stop_on_possession_loss and step.result.result == ActionResult.INTERCEPTED:
                break

            # Advance to next state
            current_state = step.state_after

        return steps

    def analyze_options(self, state: GameState) -> str:
        """
        Analyze all options for a state and return human-readable summary.
        """
        evaluated = self.evaluator.evaluate(state)

        lines = [
            "=" * 50,
            "DECISION ANALYSIS",
            "=" * 50,
            f"Ball position: ({state.ball_position.x:.1f}, {state.ball_position.y:.1f})",
            f"Ball carrier: {state.ball_carrier.id if state.ball_carrier else 'None'}",
            f"State score: {evaluated.score.total:.3f}",
            "",
            "AVAILABLE ACTIONS (ranked by expected value):",
            "-" * 50,
        ]

        for i, action in enumerate(evaluated.available_actions[:5]):
            lines.append(
                f"{i+1}. {action.action_type.value:15} "
                f"→ ({action.target.x:.1f}, {action.target.y:.1f}) "
                f"| P(success)={action.success_probability:.2f} "
                f"| EV={action.expected_value:.3f}"
            )

        if evaluated.available_actions:
            best = evaluated.available_actions[0]
            lines.extend([
                "",
                "-" * 50,
                f"RECOMMENDED: {best.action_type.value}",
                f"Target: ({best.target.x:.1f}, {best.target.y:.1f})",
                f"Success probability: {best.success_probability:.1%}",
                f"Expected value: {best.expected_value:.3f}",
            ])

        return "\n".join(lines)


def create_test_scenario() -> GameState:
    """
    Create a simple test scenario for the decision engine.

    Scenario: Build-up play from defensive third against a mid-block.
    """
    # Attacking team (building from back)
    attackers = [
        Player(id="GK", position=Position(-45, 0), team="attack"),
        Player(id="CB1", position=Position(-35, -15), team="attack"),
        Player(id="CB2", position=Position(-35, 15), team="attack"),
        Player(id="LB", position=Position(-30, -30), team="attack"),
        Player(id="RB", position=Position(-30, 30), team="attack"),
        Player(id="CM1", position=Position(-15, -10), team="attack"),
        Player(id="CM2", position=Position(-15, 10), team="attack"),
        Player(id="LW", position=Position(5, -25), team="attack"),
        Player(id="RW", position=Position(5, 25), team="attack"),
        Player(id="ST1", position=Position(15, -5), team="attack"),
        Player(id="ST2", position=Position(15, 5), team="attack"),
    ]

    # Defending team (mid-block)
    defenders = [
        Player(id="D_GK", position=Position(45, 0), team="defense"),
        Player(id="D_CB1", position=Position(30, -10), team="defense"),
        Player(id="D_CB2", position=Position(30, 10), team="defense"),
        Player(id="D_LB", position=Position(25, -25), team="defense"),
        Player(id="D_RB", position=Position(25, 25), team="defense"),
        Player(id="D_CM1", position=Position(10, -15), team="defense"),
        Player(id="D_CM2", position=Position(10, 0), team="defense"),
        Player(id="D_CM3", position=Position(10, 15), team="defense"),
        Player(id="D_LW", position=Position(-5, -20), team="defense"),
        Player(id="D_RW", position=Position(-5, 20), team="defense"),
        Player(id="D_ST", position=Position(-15, 0), team="defense"),
    ]

    # Ball starts with CB1
    ball_carrier = attackers[1]  # CB1

    return GameState(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        attackers=attackers,
        defenders=defenders,
        timestamp=0.0
    )


def run_demo():
    """Run a demonstration of the decision engine."""
    print("=" * 60)
    print("FOOTBALL DECISION ENGINE DEMO")
    print("=" * 60)
    print()

    # Create scenario
    state = create_test_scenario()

    # Create engine
    engine = DecisionEngine()

    # Analyze initial state
    print(engine.analyze_options(state))
    print()

    # Run simulation
    print("=" * 60)
    print("RUNNING SIMULATION")
    print("=" * 60)

    steps = engine.run(state, max_steps=10)

    for i, step in enumerate(steps):
        print(f"\nStep {i+1}:")
        if step.action_chosen:
            print(f"  Action: {step.action_chosen.action_type.value}")
            print(f"  Target: ({step.action_chosen.target.x:.1f}, {step.action_chosen.target.y:.1f})")
            print(f"  Result: {step.result.result.value}")
            print(f"  {step.result.message}")
        else:
            print("  No action taken")

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
