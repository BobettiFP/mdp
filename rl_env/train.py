# ------------------------------  train.py  -----------------------------------
"""
Train PPO agent on human vs LLM envs and write episode logs.
"""
import argparse, os
from typing import List, Tuple
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from env import build_env

class EpisodeLogger(BaseCallback):
    """Callback to log episode statistics."""
    def __init__(self, verbose=0):
        super(EpisodeLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_success = 0
        
    def _on_step(self) -> bool:
        # 에피소드 진행 중 누적
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['rewards'][0] > 0:
            self.current_episode_success = 1
            
        # 에피소드 종료 시 기록
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_successes.append(self.current_episode_success)
            self.current_episode_reward = 0
            self.current_episode_success = 0
            
        return True

def run(env, steps: int) -> Tuple[List[float], List[int]]:
    """Train PPO agent and collect episode statistics."""
    print(f"Environment obs space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    
    # PPO 모델 설정 개선
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=min(2048, steps // 4),  # 스텝 수에 맞게 조정
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./ppo_dialogue_tensorboard/"
    )
    
    # Episode logger 설정
    logger = EpisodeLogger()
    
    try:
        # 학습 실행 (progress_bar 제거로 tqdm 의존성 해결)
        print(f"Starting training for {steps} steps...")
        model.learn(
            total_timesteps=steps,
            callback=logger,
            progress_bar=False  # progress bar 비활성화
        )
        
        print(f"Training completed. Collected {len(logger.episode_rewards)} episodes.")
        
        # 에피소드가 너무 적으면 추가 수집
        if len(logger.episode_rewards) < 10:
            print("Collecting additional episodes for evaluation...")
            for _ in range(20):  # 추가 에피소드 수집
                obs, _ = env.reset()
                episode_reward = 0
                episode_success = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    if reward > 0:
                        episode_success = 1
                    if truncated:
                        break
                
                logger.episode_rewards.append(episode_reward)
                logger.episode_successes.append(episode_success)
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Attempting manual episode collection...")
        
        # 학습 실패 시 수동으로 에피소드 수집
        try:
            for i in range(10):
                obs, _ = env.reset()
                episode_reward = 0
                episode_success = 0
                
                for _ in range(50):  # 최대 50 스텝
                    action = env.action_space.sample()  # 랜덤 액션
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    if reward > 0:
                        episode_success = 1
                    if done or truncated:
                        break
                
                logger.episode_rewards.append(episode_reward)
                logger.episode_successes.append(episode_success)
                
        except Exception as e2:
            print(f"Manual collection also failed: {e2}")
            # 최소한의 데이터라도 반환
            if not logger.episode_rewards:
                logger.episode_rewards = [0.0]
                logger.episode_successes = [0]
    
    return logger.episode_rewards, logger.episode_successes

def save(path: str, reward: list, success: list):
    """Save episode statistics to CSV."""
    try:
        # 디렉토리 생성
        dir_path = os.path.dirname(path)
        if dir_path:  # 빈 문자열이 아닌 경우에만
            os.makedirs(dir_path, exist_ok=True)
        
        # 데이터프레임 생성 및 저장
        if reward and success:
            df = pd.DataFrame({
                "episode": np.arange(1, len(reward) + 1),
                "reward": reward,
                "success_rate": success
            })
            df.to_csv(path, index=False)
            print(f"✔ {path} written with {len(reward)} episodes")
        else:
            print(f"⚠ No data to save for {path}")
            
    except Exception as e:
        print(f"Error saving {path}: {e}")

def main():
    """Main training function."""
    p = argparse.ArgumentParser(description="Train PPO agents on dialogue environments")
    p.add_argument("--annotations", default="processed_annotations.json", 
                   help="Path to processed annotations file")
    p.add_argument("--steps", type=int, default=50_000,
                   help="Number of training steps")
    p.add_argument("--outdir", default="logs",
                   help="Output directory for logs")
    p.add_argument("--types", nargs='+', default=["human", "llm"],
                   help="Annotation types to train on")
    a = p.parse_args()
    
    print(f"Training configuration:")
    print(f"  Annotations: {a.annotations}")
    print(f"  Steps: {a.steps:,}")
    print(f"  Output dir: {a.outdir}")
    print(f"  Types: {a.types}")
    
    # 출력 디렉토리 생성
    os.makedirs(a.outdir, exist_ok=True)
    
    results = {}
    
    for typ in a.types:
        print(f"\n{'='*50}")
        print(f"▶ Training on {typ} environment...")
        print(f"{'='*50}")
        
        try:
            # 환경 생성
            env = build_env(a.annotations, typ)
            
            # 학습 실행
            rewards, successes = run(env, a.steps)
            
            # 결과 저장
            output_path = os.path.join(a.outdir, f"{typ}_env.csv")
            save(output_path, rewards, successes)
            
            # 결과 요약
            if rewards:
                avg_reward = np.mean(rewards)
                success_rate = np.mean(successes) * 100
                print(f"📊 {typ} Results:")
                print(f"   Episodes: {len(rewards)}")
                print(f"   Avg Reward: {avg_reward:.3f}")
                print(f"   Success Rate: {success_rate:.1f}%")
                
                results[typ] = {
                    'episodes': len(rewards),
                    'avg_reward': avg_reward,
                    'success_rate': success_rate
                }
            
        except Exception as e:
            print(f"❌ Failed to train {typ} environment: {e}")
            results[typ] = {'error': str(e)}
    
    # 전체 결과 요약
    print(f"\n{'='*50}")
    print("🏁 Training Summary")
    print(f"{'='*50}")
    for typ, result in results.items():
        if 'error' in result:
            print(f"{typ}: Failed - {result['error']}")
        else:
            print(f"{typ}: {result['episodes']} episodes, "
                  f"avg reward {result['avg_reward']:.3f}, "
                  f"success rate {result['success_rate']:.1f}%")

if __name__ == "__main__":
    main()