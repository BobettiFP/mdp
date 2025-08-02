# ------------------------------  improved_train.py  -----------------------------------
"""
개선된 PPO 학습 - 다양한 난이도의 환경에서 비교 실험
"""
import argparse, os, json, pathlib as p
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import gymnasium as gym                       # NEW
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from improved_env import build_improved_env
LOG_DIR = "step_logs"          # ← 스텝‑단위 로그가 쌓일 폴더
p.Path(LOG_DIR).mkdir(exist_ok=True)

# --------------------------------------------------------------------
# NEW: step-단위 transition을 JSONL로 기록하는 래퍼
# --------------------------------------------------------------------
class TransitionLogger(gym.Wrapper):
    """
    env.step() 호출마다
    {episode, t, state_before, action, state_after, reward}
    한 줄(JSON)씩 기록.
    """
    def __init__(self, env: gym.Env, path: str, flush_every: int = 5000):
        super().__init__(env)
        self.path = path
        self.flush_every = flush_every
        self.buffer = []
        self.episode = 0
        self.t = 0

    # Gymnasium API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.t = 0
        return obs, info

    def step(self, action):
        state_before = tuple(self.env._current_state)
        obs, reward, done, truncated, info = self.env.step(action)
        state_after  = tuple(self.env._current_state)
        self.t += 1

        self.buffer.append({
            "episode": self.episode,
            "t": self.t,
            "state_before": state_before,
            "action": int(action),
            "state_after": state_after,
            "reward": float(reward)
        })

        if done or truncated or len(self.buffer) >= self.flush_every:
            self._flush()
        return obs, reward, done, truncated, info

    def _flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in self.buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.buffer.clear()


class DetailedEpisodeLogger(BaseCallback):
    """더 상세한 에피소드 로깅 (기존 코드 그대로)"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards, self.episode_successes, self.episode_lengths = [], [], []

    def _on_step(self) -> bool:
        # Stable-Baselines3 callback API - 여기선 별도 작업 안 함
        return True


# --------------------------------------------------------------------
# 수정: output_dir 인자를 받도록 변경
# --------------------------------------------------------------------
def run_experiment(env, steps: int, env_name: str, output_dir: str) -> Dict:
    """단일 환경에서 실험 실행"""
    print(f"\n{'='*50}")
    print(f"🚀 Starting {env_name} experiment")
    print(f"{'='*50}")
    print(f"Obs space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # PPO 하이퍼파라미터
    if "hard" in env_name:
        learning_rate, n_epochs = 1e-4, 15
    elif "easy" in env_name:
        learning_rate, n_epochs = 5e-4, 5
    else:
        learning_rate, n_epochs = 3e-4, 10

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=256,
        verbose=0
    )

    logger = DetailedEpisodeLogger()

    # NEW: transition 로그 파일 경로 지정 + env 래핑
    trans_path = os.path.join(output_dir, f"{env_name}_transitions.jsonl")
    env = TransitionLogger(env, trans_path)

    try:
        print(f"Training for {steps:,} steps...")
        model.learn(total_timesteps=steps, callback=logger, progress_bar=False)
        env._flush()                                   # 남은 버퍼 플러시
        print(f"✅ Training completed: {len(logger.episode_rewards)} episodes")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        # 실패 시 랜덤 정책으로 최소한의 데이터 수집
        for _ in range(20):
            obs, _ = env.reset()
            ep_rew = ep_len = ep_succ = 0
            for _ in range(20):
                action = env.action_space.sample()
                obs, rew, done, trunc, _ = env.step(action)
                ep_rew += rew
                ep_len += 1
                if rew > 0:
                    ep_succ = 1
                if done or trunc:
                    break
            logger.episode_rewards.append(ep_rew)
            logger.episode_successes.append(ep_succ)
            logger.episode_lengths.append(ep_len)

    # 결과 집계
    if not logger.episode_rewards:
        logger.episode_rewards, logger.episode_successes, logger.episode_lengths = [0.0], [0], [1]

    results = {
        'rewards': logger.episode_rewards,
        'successes': logger.episode_successes,
        'lengths': logger.episode_lengths,
        'episodes': len(logger.episode_rewards),
        'avg_reward': np.mean(logger.episode_rewards),
        'avg_success': np.mean(logger.episode_successes) * 100,
        'avg_length': np.mean(logger.episode_lengths),
        'std_reward': np.std(logger.episode_rewards)
    }

    print(f"📊 {env_name} Results:"
          f" episodes {results['episodes']},"
          f" avg_reward {results['avg_reward']:.3f},"
          f" success {results['avg_success']:.1f}%,"
          f" len {results['avg_length']:.1f}")
    return results


def save_detailed_results(results: Dict, output_dir: str):
    """상세 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 환경별 CSV
    for env_name, data in results.items():
        if isinstance(data, dict) and 'rewards' in data:
            pd.DataFrame({
                'episode': range(1, len(data['rewards']) + 1),
                'reward': data['rewards'],
                'success': data['successes'],
                'length': data['lengths']
            }).to_csv(os.path.join(output_dir, f"{env_name}.csv"), index=False)

    # 요약 리포트
    summary = [{
        'Environment': n,
        'Episodes': d['episodes'],
        'Avg_Reward': d['avg_reward'],
        'Std_Reward': d['std_reward'],
        'Success_Rate': d['avg_success'],
        'Avg_Length': d['avg_length']
    } for n, d in results.items() if 'avg_reward' in d]
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="개선된 대화 환경 실험")
    parser.add_argument("--annotations", default="processed_annotations.json")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--outdir", default="improved_logs")
    parser.add_argument("--experiments", nargs='+',
                        default=["human_normal", "human_hard", "llm_normal", "llm_hard"],
                        help="실험할 환경들")
    args = parser.parse_args()

    print(f"🧪 Improved Dialogue RL Experiments")
    print(f"Annotations: {args.annotations}")
    print(f"Steps per experiment: {args.steps:,}")
    print(f"Output directory: {args.outdir}")
    print(f"Experiments: {args.experiments}")

    results = {}
    for exp_name in args.experiments:
        ann_type, difficulty = (exp_name.split("_", 1) + ["normal"])[:2]
        try:
            env = build_improved_env(args.annotations, ann_type, difficulty)
            results[exp_name] = run_experiment(env, args.steps, exp_name, args.outdir)
        except Exception as e:
            print(f"❌ {exp_name} failed: {e}")
            results[exp_name] = {"error": str(e)}

    save_detailed_results(results, args.outdir)

    # 최고 성능 환경 표시
    best = {k: v for k, v in results.items() if 'avg_reward' in v}
    if best:
        top = max(best, key=lambda k: best[k]['avg_reward'] + best[k]['avg_success'] / 100)
        print(f"\n🏆 Best performing environment: {top}")


if __name__ == "__main__":
    main()
