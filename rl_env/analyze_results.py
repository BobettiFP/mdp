#!/usr/bin/env python3
"""
범용 강화학습 결과 분석 및 시각화 스크립트
자동으로 CSV 파일들을 감지하고 컬럼을 매핑합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RLAnalyzer:
    """강화학습 결과 분석기"""
    
    def __init__(self, config: dict = None):
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # 컬럼 매핑 설정
        self.column_mappings = self.config.get('column_mappings', {})
        self.results = {}
        
    def _load_default_config(self) -> dict:
        """기본 설정을 로드합니다."""
        return {
            'file_patterns': ['*.csv'],  # 모든 CSV 파일
            'column_mappings': {
                'episode': ['episode', 'ep', 'step', 'iteration', 'iter', 'episodes'],
                'reward': ['reward', 'total_reward', 'cumulative_reward', 'return', 'avg_reward'],
                'success': ['success', 'done', 'win', 'complete', 'solved', 'success_rate'],
                'length': ['length', 'steps', 'duration', 'time', 'avg_length'],
                'environment': ['environment', 'env', 'env_type', 'type']
            },
            'success_rate_window': 'auto',  # 'auto' or integer
            'plot_style': 'seaborn',
            'language': 'auto',  # 'auto', 'ko', 'en'
            'output_format': ['plot', 'report', 'summary'],
            'plot_config': {
                'figsize': (15, 10),
                'dpi': 300,
                'style': 'default'  # 더 안전한 기본값
            }
        }
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """컬럼을 자동으로 감지하고 매핑합니다."""
        detected = {}
        df_columns_lower = [col.lower() for col in df.columns]
        
        for target_col, possible_names in self.column_mappings.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    # 실제 컬럼명 찾기
                    actual_col = df.columns[df_columns_lower.index(possible_name.lower())]
                    detected[target_col] = actual_col
                    break
        
        return detected
    
    def calculate_success_rate(self, df: pd.DataFrame, success_col: str) -> pd.Series:
        """성공률을 계산합니다."""
        if self.config['success_rate_window'] == 'auto':
            window = max(1, min(100, len(df) // 20))
        else:
            window = self.config['success_rate_window']
        
        return df[success_col].rolling(window=window, min_periods=1).mean()
    
    def load_results(self, log_dir: str) -> Dict[str, pd.DataFrame]:
        """로그 디렉토리에서 결과 파일들을 자동으로 로드합니다."""
        results = {}
        log_path = Path(log_dir)
        
        # 모든 CSV 파일 찾기
        csv_files = []
        for pattern in self.config['file_patterns']:
            csv_files.extend(log_path.glob(pattern))
        
        print(f"🔍 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            env_type = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                
                # 컬럼 자동 감지
                column_map = self.auto_detect_columns(df)
                
                # 특별 처리: experiment_summary 같은 환경별 요약 파일
                if 'environment' in column_map and len(df) <= 10:  # 요약 파일로 추정
                    print(f"📋 {csv_file}: Detected as summary file")
                    # 요약 파일은 별도 처리하거나 스킵
                    continue
                
                if not column_map:
                    print(f"⚠ {csv_file}: No recognizable columns found")
                    print(f"  Available columns: {list(df.columns)}")
                    # 다른 형태의 컬럼들도 체크
                    if any(col.lower() in ['environment', 'env'] for col in df.columns):
                        print(f"  💡 Looks like a summary file - skipping")
                    continue
                
                # 필수 컬럼 확인
                if 'reward' not in column_map:
                    print(f"⚠ {csv_file}: No reward column found")
                    continue
                
                # 컬럼명 표준화
                renamed_df = df.copy()
                for standard_name, actual_name in column_map.items():
                    if actual_name != standard_name:
                        renamed_df = renamed_df.rename(columns={actual_name: standard_name})
                
                # episode 컬럼이 없으면 생성
                if 'episode' not in column_map:
                    renamed_df['episode'] = range(1, len(renamed_df) + 1)
                
                # success_rate 계산 (success 컬럼이 있는 경우)
                if 'success' in column_map:
                    renamed_df['success_rate'] = self.calculate_success_rate(
                        renamed_df, 'success'
                    )
                
                results[env_type] = renamed_df
                print(f"✔ {env_type}: {len(renamed_df)} episodes loaded")
                print(f"  Detected columns: {column_map}")
                
            except Exception as e:
                print(f"❌ Failed to load {csv_file}: {e}")
        
        self.results = results
        return results
    
    def plot_learning_curves(self, save_path: str = None):
        """학습 곡선을 플롯합니다."""
        if not self.results:
            print("No data to plot")
            return
        
        # 동적으로 서브플롯 개수 결정
        has_success = any('success' in df.columns for df in self.results.values())
        has_length = any('length' in df.columns for df in self.results.values())
        
        subplot_configs = [('reward', 'Reward')]
        if has_success:
            subplot_configs.append(('success_rate', 'Success Rate'))
        if has_length:
            subplot_configs.append(('length', 'Episode Length'))
        
        # 추가 분석
        subplot_configs.append(('reward_dist', 'Reward Distribution'))
        
        n_plots = len(subplot_configs)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        # 플롯 스타일 안전하게 설정
        try:
            if 'seaborn' in plt.style.available:
                plt.style.use('seaborn-v0_8')
            elif 'ggplot' in plt.style.available:
                plt.style.use('ggplot')
            else:
                plt.style.use('default')
        except:
            plt.style.use('default')
            
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=self.config['plot_config']['figsize'])
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle("Learning Analysis", fontsize=16)
        
        for idx, (metric, title) in enumerate(subplot_configs):
            ax = axes[idx]
            
            if metric == 'reward_dist':
                # 보상 분포 히스토그램
                for env_type, df in self.results.items():
                    if len(df) >= 2:
                        ax.hist(df['reward'], alpha=0.6, label=env_type, 
                               bins=30, density=True)
                ax.set_xlabel('Reward')
                ax.set_ylabel('Density')
            else:
                # 시계열 플롯
                for env_type, df in self.results.items():
                    if len(df) < 2 or metric not in df.columns:
                        continue
                    
                    # 원본 데이터 (투명)
                    ax.plot(df['episode'], df[metric], alpha=0.3, 
                           label=f'{env_type} (raw)')
                    
                    # 평활화된 데이터
                    window = max(1, len(df) // 50)
                    if window > 1:
                        smoothed = df[metric].rolling(window=window, min_periods=1).mean()
                        ax.plot(df['episode'], smoothed, linewidth=2, 
                               label=f'{env_type} (smooth)')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel(title)
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 빈 서브플롯 숨기기
        for idx in range(len(subplot_configs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plot_config']['dpi'], 
                       bbox_inches='tight')
            print(f"✔ Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: str = None) -> str:
        """상세 분석 리포트를 생성합니다."""
        if not self.results:
            return "No data available for analysis"
        
        # 언어 감지
        is_korean = any('한국' in str(df.columns) for df in self.results.values()) or \
                   self.config['language'] == 'ko'
        
        if is_korean:
            report = self._generate_korean_report()
        else:
            report = self._generate_english_report()
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✔ Report saved to {output_path}")
        
        return report
    
    def _generate_english_report(self) -> str:
        """영어 리포트 생성"""
        report = []
        report.append("# Reinforcement Learning Results Analysis Report\n")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        total_episodes = sum(len(df) for df in self.results.values())
        report.append("## Summary\n")
        report.append(f"- Total episodes: {total_episodes:,}")
        report.append(f"- Environments: {len(self.results)}")
        report.append(f"- Analyzed: {', '.join(self.results.keys())}\n")
        
        # Environment analysis
        report.append("## Environment Analysis\n")
        
        for env_type, df in self.results.items():
            if len(df) == 0:
                continue
            
            report.append(f"### {env_type.upper()}\n")
            
            # Basic stats
            reward_stats = df['reward'].describe()
            
            report.append("#### Basic Statistics")
            report.append(f"- Episodes: {len(df):,}")
            report.append(f"- Mean reward: {reward_stats['mean']:.3f}")
            report.append(f"- Reward std: {reward_stats['std']:.3f}")
            report.append(f"- Max reward: {reward_stats['max']:.3f}")
            report.append(f"- Min reward: {reward_stats['min']:.3f}")
            
            if 'success' in df.columns:
                success_rate = df['success'].mean() * 100
                report.append(f"- Success rate: {success_rate:.1f}%")
            
            if 'length' in df.columns:
                length_stats = df['length'].describe()
                report.append(f"- Mean episode length: {length_stats['mean']:.2f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _generate_korean_report(self) -> str:
        """한국어 리포트 생성"""
        report = []
        report.append("# 강화학습 결과 분석 리포트\n")
        report.append(f"생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 전체 요약
        total_episodes = sum(len(df) for df in self.results.values())
        report.append("## 전체 요약\n")
        report.append(f"- 총 에피소드 수: {total_episodes:,}")
        report.append(f"- 환경 수: {len(self.results)}")
        report.append(f"- 분석 환경: {', '.join(self.results.keys())}\n")
        
        # 환경별 분석
        report.append("## 환경별 분석\n")
        
        for env_type, df in self.results.items():
            if len(df) == 0:
                continue
            
            report.append(f"### {env_type.upper()}\n")
            
            # 기본 통계
            reward_stats = df['reward'].describe()
            
            report.append("#### 기본 통계")
            report.append(f"- 에피소드 수: {len(df):,}")
            report.append(f"- 평균 보상: {reward_stats['mean']:.3f}")
            report.append(f"- 보상 표준편차: {reward_stats['std']:.3f}")
            report.append(f"- 최대 보상: {reward_stats['max']:.3f}")
            report.append(f"- 최소 보상: {reward_stats['min']:.3f}")
            
            if 'success' in df.columns:
                success_rate = df['success'].mean() * 100
                report.append(f"- 성공률: {success_rate:.1f}%")
            
            if 'length' in df.columns:
                length_stats = df['length'].describe()
                report.append(f"- 평균 에피소드 길이: {length_stats['mean']:.2f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def print_summary(self):
        """콘솔에 요약 정보를 출력합니다."""
        if not self.results:
            print("No data available")
            return
        
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        for env_type, df in self.results.items():
            if len(df) > 0:
                avg_reward = df['reward'].mean()
                
                summary_parts = [
                    f"{env_type:>15}: {len(df):>6} episodes",
                    f"avg reward {avg_reward:>6.3f}"
                ]
                
                if 'success' in df.columns:
                    success_rate = df['success'].mean() * 100
                    summary_parts.append(f"success {success_rate:>5.1f}%")
                
                if 'length' in df.columns:
                    avg_length = df['length'].mean()
                    summary_parts.append(f"length {avg_length:>5.1f}")
                
                print(", ".join(summary_parts))

def load_config_file(config_path: str) -> dict:
    """설정 파일을 로드합니다."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Unsupported config format: {config_path.suffix}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="범용 강화학습 결과 분석기")
    parser.add_argument("--logdir", default=".", help="로그 디렉토리 경로")
    parser.add_argument("--output", default="analysis", help="출력 파일 prefix")
    parser.add_argument("--config", help="설정 파일 경로 (JSON/YAML)")
    parser.add_argument("--no-plot", action="store_true", help="플롯 생성 건너뛰기")
    parser.add_argument("--pattern", action="append", help="파일 패턴 추가")
    parser.add_argument("--window", type=int, help="성공률 계산 윈도우 크기")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = {}
    if args.config:
        config = load_config_file(args.config)
    
    # CLI 인수로 설정 오버라이드
    if args.pattern:
        config['file_patterns'] = args.pattern
    if args.window:
        config['success_rate_window'] = args.window
    
    # 분석기 초기화
    analyzer = RLAnalyzer(config)
    
    print(f"📊 Analyzing results from {args.logdir}...")
    print(f"🔍 File patterns: {analyzer.config['file_patterns']}")
    
    # 결과 로드
    results = analyzer.load_results(args.logdir)
    
    if not results:
        print("❌ No analyzable results found!")
        print("💡 Tips:")
        print("  - Check file patterns in config")
        print("  - Ensure CSV files have recognizable column names")
        print("  - Use --pattern to specify custom patterns")
        return
    
    # 플롯 생성
    if not args.no_plot:
        try:
            analyzer.plot_learning_curves(f"{args.output}_plots.png")
        except ImportError:
            print("⚠ matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠ Plot generation failed: {e}")
    
    # 리포트 생성
    analyzer.generate_report(f"{args.output}_report.md")
    
    # 요약 출력
    analyzer.print_summary()

if __name__ == "__main__":
    main()