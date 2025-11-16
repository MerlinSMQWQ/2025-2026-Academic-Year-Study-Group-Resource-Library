# 说明
因为我已经将uv相关的配置文件放进来了，所以到当前目录下，使用uv sync即可

```bash
cd RL/value-based
uv sync
```

# 快速搭建环境
基本已经了解了强化学习的一些基础知识，我就要开始介绍我们要怎么实现强化学习的一些算法。
首先就是，我们知道我们在强化学习中非常看重环境，那我们现在就来介绍一下强化学习的环境框架：gym（gymnasium），我更推荐gymnasium，因为他是重构版本的gym，并且现在有比较好的文档支持。Gym是OpenAI的，而gymnasium是他的一个维护分支，现在也是做强化学习用的最多的一个环境框架，他的接口足够简单，然后内置了一些简单地游戏，可以很方便地测试我们的强化学习算法。
现在我们来快速搭建一个测试的游戏环境：

```python
import gymnasium as gym

# 指定构建的游戏环境是“CliffWalking-v1”，渲染的模式是认为可见的模式，没执行一次都可以用人类可以理解的方式呈现到屏幕上（简单可视化，基于pygame）
env = gym.make("CliffWalking-v1", render_mode = "human")
# observation是现在观察到的状态，info是详细信息
observation, info = env.reset()
print(f"Start observation: {observation}")

episode_over = False
total_reward: float = 0

while not episode_over:
    env.render()
    action = env.action_space.sample()
    # terminated是说任务是否正常结束，truncated是说任务是否被截断（我们可以设置智能体最多执行的步数，超过就截断）
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    episode_over = terminated or truncated
    
print(f"Episode finished! Total reward: {total_reward}")

env.close()
```