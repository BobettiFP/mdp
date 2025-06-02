from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dialogue_training')

# 훈련 중 로깅
writer.add_scalar('Reward/Episode', episode_reward, episode)
writer.add_scalar('Length/Episode', episode_length, episode)

# 네트워크 가중치 히스토그램
for name, param in self.agent.policy_net.named_parameters():
    writer.add_histogram(name, param, episode)