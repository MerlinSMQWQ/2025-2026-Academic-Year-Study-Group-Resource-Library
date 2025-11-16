# 导入所需的库
from collections import defaultdict  # 用于创建默认字典
from sys import float_info          # 系统浮点信息
import gymnasium as gym             # 强化学习环境库
import numpy as np                  # 数值计算库
from typing import DefaultDict, Tuple  # 类型提示
from tqdm import tqdm               # 进度条显示
from matplotlib import pyplot as plt  # 数据可视化

class CliffWalkingSarsaAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ) -> None:
        """初始化SARSA智能体参数

        Args:
            env: 智能体所处的环境
            learning_rate: 学习率(α)，控制更新步长
            initial_epsilon: 初始的epsilon值，用于ε-贪婪策略
            epsilon_decay: ε衰减量，每回合减少的值
            final_epsilon: 最终的epsilon值，不会低于此值
            discount_factor: 折扣因子(γ)，未来奖励的重要程度
        """
        # 保存环境引用
        self.env = env
        # 确保动作空间是离散的
        assert isinstance(env.action_space, gym.spaces.Discrete), "Only supports discrete action spaces."
        
        # 获取动作空间大小
        action_space_n: int = env.action_space.n  # type: ignore

        # 初始化Q值表: state -> [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
        # 对于每个状态，默认初始化为4个动作的零值数组
        self.q_values: DefaultDict[Tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(action_space_n)
        )

        # 设置算法参数
        self.lr = learning_rate          # 学习率 α
        self.discount_factor = discount_factor  # 折扣因子 γ
        self.epsilon = initial_epsilon   # 探索率 ε
        self.epsilon_decay = epsilon_decay     # ε衰减量
        self.final_epsilon = final_epsilon     # 最小ε值
        self.training_error: list[float] = []  # 记录训练误差

    def get_action(self, obs: tuple[int, int]) -> int:
        """
        使用ε-贪婪策略选择动作
        - 以ε概率随机探索
        - 以(1-ε)概率选择当前最优动作
        
        Args:
            obs: 当前观察到的状态(observation)
            
        Returns:
            选择的动作编号(0-3)
        """
        # 生成[0,1)之间的随机数，与ε比较决定策略
        if np.random.random() < self.epsilon:
            # 探索：随机选择一个动作
            return self.env.action_space.sample()
        else:
            # 利用：选择Q值最大的动作
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int],      # 当前状态 s
        action: int,              # 当前动作 a
        reward: float,            # 获得的奖励 r
        terminated: bool,         # 是否终止
        next_obs: tuple[int, int], # 下一状态 s'
        next_action: int,         # 下一动作 a' (SARSA的关键)
    ):
        """
        SARSA更新规则: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        这是on-policy算法，使用下一状态实际选择的动作对应的Q值
        """
        # 如果不是终止状态，则计算未来Q值；否则为0
        # 注意：这里使用的是实际选择的下一个动作next_action，而不是最大Q值
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        
        # 计算目标值: r + γ * Q(s',a')
        target = reward + self.discount_factor * future_q_value

        # 计算时序差分误差: target - Q(s,a)
        temporal_difference = target - self.q_values[obs][action]

        # 更新Q值: Q(s,a) ← Q(s,a) + α * TD_error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # 记录训练误差，用于后续分析
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        衰减探索率ε，使其逐渐减少探索，增加利用
        """
        # 确保ε不会低于设定的最小值
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def render_agent_demo(agent, env, num_episodes=3):
    """实时渲染智能体在环境中的表现，用于可视化演示
    
    Args:
        agent: 训练好的智能体
        env: 带渲染功能的环境
        num_episodes: 演示回合数
    """
    # 保存原始 epsilon 值，演示结束后恢复
    old_epsilon = agent.epsilon
    # 关闭探索，只利用已学到的策略进行演示
    agent.epsilon = 0.0
    
    print("开始演示智能体...")
    print("说明: 智能体会执行3个完整的回合")
    print("观看窗口中：")
    print("- 红色方块: 智能体当前位置")
    print("- 蓝色区域: 起始位置") 
    print("- 白色区域: 安全区域")
    print("- 黑色区域: 悬崖(掉下去会失败)")
    print("- 绿色区域: 目标位置")
    print("- 按任意键继续到下一步，关闭窗口结束演示\n")
    
    try:
        # 进行多个回合的演示
        for episode in range(num_episodes):
            # 重置环境并获取初始观察
            obs, info = env.reset()
            # CliffWalking环境返回的是整数状态，需要转换为坐标形式(row, col)
            obs = (obs // 12, obs % 12)  # 转换为坐标
            
            print(f"第 {episode + 1} 回合开始")
            env.render()  # 显示初始状态
            
            # 回合内循环
            done = False
            step_count = 0
            total_reward = 0
            
            while not done:
                # 根据当前策略选择动作
                action = agent.get_action(obs)
                # 执行动作，获取环境反馈
                next_obs, reward, terminated, truncated, info = env.step(action)
                # 将下一状态转换为坐标形式
                next_obs = (next_obs // 12, next_obs % 12)  # 转换为坐标
                
                # 更新状态和统计信息
                obs = next_obs
                done = terminated or truncated  # 回合是否结束
                total_reward += reward          # 累计奖励
                step_count += 1                 # 步数计数
                
                # 渲染当前状态，可视化智能体位置
                env.render()
                
                # 添加小延迟以便观察智能体移动
                import time
                time.sleep(0.5)
                
                # 防止无限循环的安全机制
                if step_count > 100:
                    print("达到最大步数限制，结束当前回合")
                    break
            
            print(f"第 {episode + 1} 回合结束 - 总奖励: {total_reward}, 步数: {step_count}\n")
            
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    finally:
        # 恢复原始 epsilon 值
        agent.epsilon = old_epsilon
        print("演示完成")


# 主程序入口
if __name__ == "__main__":
    # 设置超参数
    learning_rate = 0.5      # 学习率
    n_episodes = 500         # 训练回合数
    start_epsilon = 1.0      # 初始探索率
    epsilon_decay = start_epsilon / (n_episodes/ 2)  # ε衰减量
    final_epsilon = 0.1      # 最终探索率
    
    # 创建 CliffWalking 环境
    # CliffWalking-v1是一个4x12的网格世界
    # 智能体从左下角(3,0)开始，目标是右下角(3,11)
    # 中间有悬崖(3,1)到(3,10)，掉下去会获得-100奖励并回到起点
    env = gym.make("CliffWalking-v1")
    # 包装环境以记录统计数据
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    
    # 创建SARSA智能体
    agent = CliffWalkingSarsaAgent(
        env = env,
        learning_rate = learning_rate,
        initial_epsilon = start_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor=0.9  # 折扣因子
    )

    # 记录训练过程中的奖励和长度
    episode_rewards = []
    episode_lengths = []

    # 开始训练循环
    for episode in tqdm(range(n_episodes)):
        # 重置环境，获取初始状态
        obs, info = env.reset()
        # 将整数状态转换为坐标形式 (row, col)
        obs = (obs // 12, obs % 12)
        done = False
        total_reward = 0
        
        # SARSA需要先选择第一个动作（on-policy特性）
        action = agent.get_action(obs)

        # 回合内循环，直到回合结束
        while not done:
            # 执行动作，与环境交互
            next_obs, reward, terminated, truncated, info = env.step(action)
            # 将下一状态转换为坐标形式
            next_obs = (next_obs // 12, next_obs % 12)
            
            # 选择下一个动作（这是SARSA的关键区别 - on-policy）
            next_action = agent.get_action(next_obs)
            
            # 使用SARSA更新规则进行更新
            agent.update(obs, action, float(reward), terminated, next_obs, next_action)

            # 更新状态和动作（SARSA的核心特点）
            obs = next_obs
            action = next_action
            total_reward += float(reward)

            # 检查回合是否结束
            done = terminated or truncated

        # 记录本回合的统计数据
        episode_rewards.append(total_reward)
        episode_lengths.append(info.get('episode', {}).get('l', 0))
        # 衰减探索率
        agent.decay_epsilon()

    def get_moving_avgs(arr, window, convolution_mode):
        """计算移动平均值来平滑噪声数据，使趋势更清晰
        
        Args:
            arr: 原始数据数组
            window: 移动窗口大小
            convolution_mode: 卷积模式
            
        Returns:
            平滑后的数据
        """
        return np.convolve(
            np.array(arr).flatten(),    # 将输入展平为一维数组
            np.ones(window),            # 创建窗口权重数组
            mode=convolution_mode       # 卷积模式
        ) / window                      # 归一化

    # 数据可视化部分
    # 设置移动平均窗口大小
    rolling_length = 10
    # 创建包含3个子图的图表
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # 第一个子图：回合奖励变化
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        episode_rewards,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # 第二个子图：回合长度变化
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        episode_lengths,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # 第三个子图：训练误差变化
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()
    
    def test_agent(agent, env, num_episodes=100):
        """测试训练好的智能体性能，不进行学习只进行利用
        
        Args:
            agent: 待测试的智能体
            env: 测试环境
            num_episodes: 测试回合数
        """
        total_rewards = []

        # 临时禁用探索，纯粹利用学到的策略
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # 纯利用

        for _ in range(num_episodes):
            # 重置环境
            obs, info = env.reset()
            # 将观测转换为坐标形式
            obs = (obs // 12, obs % 12)
            episode_reward = 0
            done = False
            
            # 先选择第一个动作（SARSA要求）
            action = agent.get_action(obs)

            # 执行一个完整回合
            while not done:
                # 执行动作并获取反馈
                obs, reward, terminated, truncated, info = env.step(action)
                # 将观测转换为坐标形式
                obs = (obs // 12, obs % 12)
                episode_reward += reward
                done = terminated or truncated
                
                # 如果未结束，选择下一个动作
                if not done:
                    action = agent.get_action(obs)

            total_rewards.append(episode_reward)

        # 恢复原始的探索率
        agent.epsilon = old_epsilon

        # 计算测试结果统计信息
        win_rate = np.mean(np.array(total_rewards) >= -20)  # 认为奖励大于-20为成功
        average_reward = np.mean(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Success Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")

    # 测试训练好的智能体
    test_agent(agent, env)
    
    # 创建可视化演示环境并运行演示
    render_env = gym.make("CliffWalking-v1", render_mode="human")
    render_agent_demo(agent, render_env)