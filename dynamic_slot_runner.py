#!/usr/bin/env python3
"""
Dynamic Slot Generation RL 대화 시스템 실행 스크립트 (개선된 버전)
사용자의 데이터셋을 유연하게 처리할 수 있도록 개선
"""

import argparse
import os
import sys
import json

def setup_environment():
    """실행 환경 설정"""
    print("🔧 Setting up environment...")
    
    # 필요한 디렉토리 생성
    os.makedirs("dynamic_logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("✅ Environment setup complete!")
    return True

def validate_data_file(data_path: str):
    """데이터 파일 검증"""
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        print("Please make sure your data file exists.")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Data file loaded successfully!")
        print(f"   📊 Total dialogues: {len(data)}")
        
        # 데이터 구조 검증
        if not isinstance(data, list):
            print("❌ Error: Data should be a list of dialogues")
            return False
        
        if len(data) == 0:
            print("❌ Error: No dialogues found in data")
            return False
        
        # 첫 번째 대화 구조 확인
        sample_dialogue = data[0]
        if not isinstance(sample_dialogue, dict):
            print("❌ Error: Each dialogue should be a dictionary")
            return False
        
        # 턴 구조 확인
        if 'turns' in sample_dialogue:
            turns = sample_dialogue['turns']
            if isinstance(turns, list) and len(turns) > 0:
                sample_turn = turns[0]
                if 'speaker' in sample_turn and 'utterance' in sample_turn:
                    print("✅ Data structure validation passed!")
                    return True
        
        print("⚠️  Warning: Data structure might not be optimal")
        print("   Expected format: [{'turns': [{'speaker': 'USER/SYSTEM', 'utterance': '...'}]}]")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_training(data_path: str, episodes=2000, lr=1e-4):
    """훈련 실행"""
    print(f"🚀 Starting training with {data_path} for {episodes} episodes...")
    
    try:
        # 메인 시스템 import 및 실행
        from dynamic_slot_rl_system import DynamicDialogueTrainer
        
        trainer = DynamicDialogueTrainer(data_path)
        trainer.train(num_episodes=episodes)
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_evaluation(data_path: str, model_path=None):
    """평가 실행"""
    print(f"🧪 Running evaluation with {data_path}...")
    
    test_utterances = [
        "I need a luxury hotel with premium amenities and concierge services",
        "Looking for an eco-friendly restaurant with organic ingredients and sustainable practices",
        "Want a high-speed train with wifi connectivity and power outlets for work",
        "Need a premium taxi with leather seats and climate control for comfort",
        "Searching for educational attractions with interactive exhibits and guided tours"
    ]
    
    try:
        from dynamic_slot_rl_system import DynamicDialogueTrainer
        
        trainer = DynamicDialogueTrainer(data_path)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            # 모델 로딩 로직 추가 필요
        
        trainer.test_dynamic_slot_generation(test_utterances)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def analyze_results(log_dir=None):
    """결과 분석"""
    print("📊 Analyzing results...")
    
    if not log_dir:
        # 가장 최근 로그 디렉토리 찾기
        if not os.path.exists("dynamic_logs"):
            print("❌ No log directories found!")
            return
            
        log_dirs = [d for d in os.listdir("dynamic_logs") if os.path.isdir(os.path.join("dynamic_logs", d))]
        if log_dirs:
            log_dir = os.path.join("dynamic_logs", sorted(log_dirs)[-1])
        else:
            print("❌ No log directories found!")
            return
    
    try:
        from dynamic_slot_rl_system import SlotEvolutionAnalyzer
        
        analyzer = SlotEvolutionAnalyzer(log_dir)
        analyzer.analyze_slot_evolution()
        
        print(f"✅ Analysis complete! Results saved in {log_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def evaluate_dataset(data_path: str):
    """데이터셋 평가"""
    print(f"📊 Evaluating dataset: {data_path}")
    
    try:
        # 슬롯 발견 평가기 임포트 및 실행
        from slot_discovery_evaluator import SlotDiscoveryEvaluator
        
        evaluator = SlotDiscoveryEvaluator()
        stats, requirements = evaluator.generate_evaluation_report(data_path)
        
        # 평가 결과 기반 권장사항
        print(f"\n💡 Recommendations based on evaluation:")
        
        if requirements['confidence'] in ['high', 'very_high']:
            print("✅ Your dataset is excellent for dynamic slot discovery!")
            print("   Proceed with full training for best results.")
        elif requirements['confidence'] == 'medium':
            print("⚠️  Your dataset is adequate for basic slot discovery.")
            print("   Consider reducing episodes or collecting more data.")
        else:
            print("❌ Dataset might be too small for reliable results.")
            print("   Strongly recommend collecting more dialogues.")
            
        return requirements['confidence']
        
    except ImportError:
        print("⚠️  Slot discovery evaluator not available. Proceeding with training...")
        return 'unknown'
    except Exception as e:
        print(f"⚠️  Dataset evaluation failed: {e}")
        return 'unknown'

def demonstrate_dynamic_slots(data_path: str):
    """동적 슬롯 생성 데모"""
    print("🎭 Dynamic Slot Generation Demo")
    print("=" * 50)
    
    demo_utterances = [
        "I want a hotel with business-class amenities and executive lounge access",
        "Need a restaurant with private dining rooms and sommelier services", 
        "Looking for a train with first-class compartments and onboard catering",
        "Want a taxi with child safety seats and pet-friendly policies",
        "Searching for attractions with VIP tours and photography permissions"
    ]
    
    try:
        from dynamic_slot_rl_system import DynamicSlotExtractor
        
        extractor = DynamicSlotExtractor()
        
        print(f"\n🔍 Extracting dynamic slots from sample utterances:")
        
        for i, utterance in enumerate(demo_utterances, 1):
            print(f"\n{i}. '{utterance}'")
            
            slots = extractor.extract_slots_from_utterance(utterance)
            
            print("   Discovered slots:")
            for domain, domain_slots in slots.items():
                if domain_slots:
                    print(f"     {domain}: {list(domain_slots)}")
        
        print(f"\n📈 Total unique slots discovered: {extractor.get_current_slot_vocabulary_size()}")
        print("\n🏷️ All discovered slots by domain:")
        
        for domain, slots in extractor.get_all_discovered_slots().items():
            if slots:
                print(f"  {domain}: {list(slots)}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def detect_data_format(data_path: str):
    """데이터 포맷 자동 감지 및 변환 제안"""
    print(f"🔍 Analyzing data format: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            
            # MultiWOZ 스타일 검사
            if isinstance(sample, dict) and 'turns' in sample:
                print("✅ Detected MultiWOZ-style format")
                return "multiwoz"
            
            # 단순 대화 리스트 검사
            elif isinstance(sample, list):
                print("✅ Detected simple dialogue list format")
                return "simple_list"
            
            # 기타 구조
            else:
                print(f"⚠️  Unknown format. Sample structure: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
                return "unknown"
        
        return "unknown"
        
    except Exception as e:
        print(f"❌ Format detection failed: {e}")
        return "unknown"

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Dynamic Slot Generation RL Dialogue System")
    
    parser.add_argument("--mode", choices=["train", "eval", "analyze", "demo", "evaluate"], 
                       default="train", help="실행 모드")
    parser.add_argument("--data", type=str, default="test_result.json",
                       help="훈련/평가용 데이터 파일 경로")
    parser.add_argument("--episodes", type=int, default=2000, 
                       help="훈련 에피소드 수")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="학습률")
    parser.add_argument("--model_path", type=str, 
                       help="평가용 모델 경로")
    parser.add_argument("--log_dir", type=str, 
                       help="분석용 로그 디렉토리")
    parser.add_argument("--auto_adjust", action="store_true",
                       help="데이터셋 크기에 따라 자동으로 에피소드 수 조정")
    
    args = parser.parse_args()
    
    print("🌟 Dynamic Slot Generation RL Dialogue System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data}")
    
    # 환경 설정
    if not setup_environment():
        sys.exit(1)
    
    # 데이터 파일 검증 (demo 모드 제외)
    if args.mode != "demo" and args.mode != "analyze":
        if not validate_data_file(args.data):
            sys.exit(1)
        
        # 데이터 포맷 감지
        data_format = detect_data_format(args.data)
        print(f"📋 Data format: {data_format}")
    
    # 모드별 실행
    if args.mode == "evaluate":
        # 데이터셋 평가
        confidence = evaluate_dataset(args.data)
        
        # 평가 결과에 따른 자동 조정
        if args.auto_adjust:
            if confidence in ['low', 'medium']:
                args.episodes = min(args.episodes, 500)
                print(f"🔧 Auto-adjusted episodes to {args.episodes} based on dataset size")
        
    elif args.mode == "train":
        # 선택적 데이터셋 평가
        if args.auto_adjust:
            print("📊 Quick dataset evaluation...")
            confidence = evaluate_dataset(args.data)
            
            if confidence == 'low':
                args.episodes = min(args.episodes, 300)
                print(f"🔧 Reduced episodes to {args.episodes} for small dataset")
            elif confidence == 'medium':
                args.episodes = min(args.episodes, 800)
                print(f"🔧 Adjusted episodes to {args.episodes} for medium dataset")
        
        trainer = run_training(args.data, args.episodes, args.lr)
        if trainer:
            print("✅ Training completed successfully!")
            print(f"📁 Results saved in: {trainer.log_dir}")
            
    elif args.mode == "eval":
        run_evaluation(args.data, args.model_path)
        
    elif args.mode == "analyze":
        analyze_results(args.log_dir)
        
    elif args.mode == "demo":
        demonstrate_dynamic_slots(args.data)
    
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()