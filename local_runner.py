import argparse
from threading import Thread

import numpy as np
from gymnasium.wrappers import TimeLimit

from agent import Agent
from core.vendor.car_racing import CarRacing
from env_wrapper import CarEnvironment


DEFAULT_AGENT_TIMEOUT_SECONDS = 5.0
MAX_INVALID_ACTIONS = 10
NO_OP_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GAME_VARIABLES_VERSION = "variables-6"


def safe_act(agent, observation, timeout_sec=DEFAULT_AGENT_TIMEOUT_SECONDS):
    outcome = []

    def call():
        try:
            outcome.append(
                np.asarray(agent.act(observation), dtype=np.float32).reshape(-1)
            )
        except Exception:
            outcome.append(None)

    thread = Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive() or not outcome or outcome[0] is None:
        return NO_OP_ACTION.copy(), False

    action = outcome[0]
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        return NO_OP_ACTION.copy(), False
    return np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]), True


def safe_reset(agent, observation, timeout_sec=DEFAULT_AGENT_TIMEOUT_SECONDS):
    if not hasattr(agent, "reset"):
        return

    outcome = []

    def call():
        try:
            agent.reset(observation)
        except Exception as error:
            outcome.append(error)

    thread = Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError("agent.reset() timed out")
    if outcome:
        raise outcome[0]


def run_local_test(track_id, seed, max_steps, frame_skip, render_mode="human"):
    print("=== 시작: 로컬 환경 테스트 ===")

    raw_frame_budget = max_steps * frame_skip + 200
    env = CarEnvironment(
        TimeLimit(
            CarRacing(continuous=True, render_mode=render_mode),
            max_episode_steps=raw_frame_budget,
        ),
        skip_frames=frame_skip,
    )

    try:
        print("에이전트를 초기화합니다...")
        agent = Agent()

        observation, info = env.reset(
            seed=seed,
            options={"track_id": track_id},
        )
        print(track_id, "번째 트랙이 선택되었습니다.")

        start_time = env.unwrapped.t
        safe_reset(agent, observation)

        total_reward = 0.0
        steps = 0
        invalid_action_counter = 0
        done = False
        local_retire_reason = None
        terminated = False
        truncated = False

        print("시뮬레이션을 시작합니다.")
        while not done and steps < max_steps:
            action, valid = safe_act(agent, observation)
            if valid:
                invalid_action_counter = 0
            else:
                invalid_action_counter += 1
                print(
                    f"유효하지 않은 행동: "
                    f"{invalid_action_counter}/{MAX_INVALID_ACTIONS}"
                )
                if invalid_action_counter >= MAX_INVALID_ACTIONS:
                    local_retire_reason = "invalid_action"
                    break

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if steps % 100 == 0:
                print(
                    f"진행 스텝: {steps}, 누적 보상: {total_reward:.2f}, "
                    f"손상: {info['damage']:.0%}"
                )

        progress = env._calculate_progress()
        finish_time_s = env.unwrapped.finish_time_s
        finish_qualified = env.unwrapped.finish_qualified_time_s is not None
        completed = finish_time_s is not None
        lap_time_ms = round((finish_time_s - start_time) * 1000) if completed else None
        retire_reason = local_retire_reason or info.get("retire_reason")
        if not completed and retire_reason is None:
            if terminated:
                retire_reason = "off_track"
            elif truncated or steps >= max_steps:
                retire_reason = "max_steps"

        print("=== 종료: 로컬 환경 테스트 ===")
        print(f"trackId: {track_id}")
        print(f"seed: {seed}")
        print(f"variables version: {GAME_VARIABLES_VERSION}")
        print(f"simulation agent steps: {steps}")
        print(f"최종 누적 보상: {total_reward:.2f}")
        print(f"progress: {progress:.6f}")
        print(f"finish qualified: {finish_qualified}")
        print(f"finish_time_s: {finish_time_s}")
        print(f"lapTimeMs: {lap_time_ms}")
        print("FINISHED" if completed else "DNF")
        if retire_reason:
            print(f"리타이어 사유: {retire_reason}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 HAIC 공식 로컬 주행 환경")
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="GUI 창 없이 실행합니다.",
    )
    args = parser.parse_args()
    run_local_test(
        args.track_id,
        args.seed,
        args.max_steps,
        args.frame_skip,
        render_mode=None if args.no_render else "human",
    )
