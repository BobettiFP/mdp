#!/usr/bin/env python3
"""
학습 결과 분석 및 시각화 스크립트
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_results(log_dir: str):
    """로그 디렉토리에서 결과 파일들을 로드합니다."""
    results = {}
    log_path = Path(log_dir)
    
    for csv_file in log_path.glob("*_env.csv"):
        env_type = csv_file.stem.replace("_env", "")
        try:
            df = pd.read_csv(csv_file)
            results[env_type] = df
            print(f"✔ Loaded {len(df)} episodes from {csv_file}")
        except Exception as e:
            print(f"❌ Failed to load {csv_file}: {e}")
    
    return results

def plot_learning_curves(results: dict, save_path: str = None):
    """학습 곡선을 플롯합니다."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Learning Curves Comparison", fontsize=16)
    
    for env_type, df in results.items():
        if len(df) < 2:
            continue
            
        # 이동평균 계산
        window = min(10, len(df) // 4)  # 적응적 윈도우 크기
        if window > 0:
            df[f'reward_ma'] = df['reward'].rolling(window=window, min_periods=1).mean()
            df[f'success_ma'] = df['success_rate'].rolling(window=window, min_periods=1).mean()
        else:
            df[f'reward_ma'] = df['reward']
            df[f'success_ma'] = df['success_rate']
    
    # 보상 곡선
    ax = axes[0, 0]
    for env_type, df in results.items():
        if len(df) >= 2:
            ax.plot(df['episode'], df['reward'], alpha=0.3, label=f'{env_type} (raw)')
            ax.plot(df['episode'], df['reward_ma'], linewidth=2, label=f'{env_type} (smooth)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 성공률 곡선
    ax = axes[0, 1]
    for env_type, df in results.items():
        if len(df) >= 2:
            ax.plot(df['episode'], df['success_rate'], alpha=0.3, label=f'{env_type} (raw)')
            ax.plot(df['episode'], df['success_ma'], linewidth=2, label=f'{env_type} (smooth)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 보상 분포
    ax = axes[1, 0]
    for env_type, df in results.items():
        if len(df) >= 2:
            ax.hist(df['reward'], alpha=0.6, label=env_type, bins=20)
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 누적 성공률
    ax = axes[1, 1]
    for env_type, df in results.items():
        if len(df) >= 2:
            cumulative_success = df['success_rate'].expanding().mean()
            ax.plot(df['episode'], cumulative_success, linewidth=2, label=env_type)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('Cumulative Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✔ Plot saved to {save_path}")
    
    plt.show()

def generate_report(results: dict, output_path: str = None):
    """상세 분석 리포트를 생성합니다."""
    report = []
    report.append("# 대화 시스템 PPO 학습 결과 분석 리포트\n")
    report.append(f"생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 전체 요약
    report.append("## 전체 요약\n")
    total_episodes = sum(len(df) for df in results.values())
    report.append(f"- 총 에피소드 수: {total_episodes:,}")
    report.append(f"- 환경 타입 수: {len(results)}")
    report.append(f"- 분석된 환경: {', '.join(results.keys())}\n")
    
    # 환경별 상세 분석
    report.append("## 환경별 상세 분석\n")
    
    for env_type, df in results.items():
        if len(df) == 0:
            continue
            
        report.append(f"### {env_type.upper()} 환경\n")
        
        # 기본 통계
        reward_stats = df['reward'].describe()
        success_rate = df['success_rate'].mean() * 100
        
        report.append("#### 기본 통계")
        report.append(f"- 에피소드 수: {len(df):,}")
        report.append(f"- 평균 보상: {reward_stats['mean']:.3f}")
        report.append(f"- 보상 표준편차: {reward_stats['std']:.3f}")
        report.append(f"- 최대 보상: {reward_stats['max']:.3f}")
        report.append(f"- 최소 보상: {reward_stats['min']:.3f}")
        report.append(f"- 전체 성공률: {success_rate:.1f}%")
        
        # 학습 진행도 분석
        if len(df) >= 10:
            early_episodes = df.head(len(df)//3)
            late_episodes = df.tail(len(df)//3)
            
            early_reward = early_episodes['reward'].mean()
            late_reward = late_episodes['reward'].mean()
            improvement = late_reward - early_reward
            
            early_success = early_episodes['success_rate'].mean() * 100
            late_success = late_episodes['success_rate'].mean() * 100
            success_improvement = late_success - early_success
            
            report.append(f"\n#### 학습 진행도")
            report.append(f"- 초기 평균 보상: {early_reward:.3f}")
            report.append(f"- 후기 평균 보상: {late_reward:.3f}")
            report.append(f"- 보상 개선도: {improvement:+.3f}")
            report.append(f"- 초기 성공률: {early_success:.1f}%")
            report.append(f"- 후기 성공률: {late_success:.1f}%")
            report.append(f"- 성공률 개선: {success_improvement:+.1f}%")
        
        report.append("")
    
    # 환경 간 비교
    if len(results) >= 2:
        report.append("## 환경 간 비교\n")
        
        comparison_data = []
        for env_type, df in results.items():
            if len(df) > 0:
                comparison_data.append({
                    '환경': env_type,
                    '에피소드수': len(df),
                    '평균보상': df['reward'].mean(),
                    '성공률': df['success_rate'].mean() * 100,
                    '보상분산': df['reward'].var()
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            report.append("| 환경 | 에피소드수 | 평균보상 | 성공률(%) | 보상분산 |")
            report.append("|------|------------|----------|-----------|----------|")
            
            for _, row in comp_df.iterrows():
                report.append(f"| {row['환경']} | {row['에피소드수']:,} | "
                            f"{row['평균보상']:.3f} | {row['성공률']:.1f} | {row['보상분산']:.3f} |")
    
    # 권장사항
    report.append("\n## 권장사항\n")
    
    best_env = None
    best_performance = -float('inf')
    
    for env_type, df in results.items():
        if len(df) > 0:
            performance = df['reward'].mean() + df['success_rate'].mean()
            if performance > best_performance:
                best_performance = performance
                best_env = env_type
    
    if best_env:
        report.append(f"- **최고 성능 환경**: {best_env}")
    
    # 일반적인 권장사항
    for env_type, df in results.items():
        if len(df) > 0:
            avg_reward = df['reward'].mean()
            success_rate = df['success_rate'].mean() * 100
            
            if avg_reward < 0.1:
                report.append(f"- {env_type}: 보상이 낮습니다. 보상 함수나 환경 설정을 검토하세요.")
            if success_rate < 10:
                report.append(f"- {env_type}: 성공률이 낮습니다. 학습 시간을 늘리거나 하이퍼파라미터를 조정하세요.")
            if len(df) < 20:
                report.append(f"- {env_type}: 에피소드가 부족합니다. 더 긴 학습이 필요합니다.")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✔ Report saved to {output_path}")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="분석 대화 시스템 PPO 학습 결과")
    parser.add_argument("--logdir", default="logs", help="로그 디렉토리 경로")
    parser.add_argument("--output", default="analysis", help="출력 파일 prefix")
    parser.add_argument("--no-plot", action="store_true", help="플롯 생성 건너뛰기")
    
    args = parser.parse_args()
    
    print(f"📊 Analyzing results from {args.logdir}...")
    
    # 결과 로드
    results = load_results(args.logdir)
    
    if not results:
        print("❌ No results found!")
        return
    
    # 플롯 생성
    if not args.no_plot:
        try:
            plot_learning_curves(results, f"{args.output}_plots.png")
        except ImportError:
            print("⚠ matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠ Plot generation failed: {e}")
    
    # 리포트 생성
    report = generate_report(results, f"{args.output}_report.md")
    
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    for env_type, df in results.items():
        if len(df) > 0:
            avg_reward = df['reward'].mean()
            success_rate = df['success_rate'].mean() * 100
            print(f"{env_type:>10}: {len(df):>4} episodes, "
                  f"avg reward {avg_reward:>6.3f}, "
                  f"success rate {success_rate:>5.1f}%")

if __name__ == "__main__":
    main()