# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch    # 导入pytorch库
import torch.nn as nn   # 导入 PyTorch 神经网络模块
import torch.optim as optim # 导入 PyTorch 优化模块

from rsl_rl.modules import ActorCriticTransformer   # 从 rsl_rl.modules 导入 ActorCriticTransformer 类
from rsl_rl.storage import RolloutStorage    # 从 rsl_rl.storage 导入 RolloutStorage 类

class PPO:  # 定义 PPO 类
    actor_critic: ActorCriticTransformer   
    def __init__(self,          
                 actor_critic, 
                 num_learning_epochs=1, 
                 num_mini_batches=1,    
                 clip_param=0.2,        
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):# 初始化存储
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):    # 定义测试模式函数
        self.actor_critic.test()    # 设置 actor_critic 为测试模式
    
    def train_mode(self):   # 定义训练模式函数
        self.actor_critic.train()   # 定义训练模式函数

    def act(self, obs, critic_obs):          # 定义 act 函数
        if self.actor_critic.is_recurrent:   # 如果 actor_critic 是循环的
            self.transition.hidden_states = self.actor_critic.get_hidden_states()  # 获取隐藏状态
 
        # Compute the actions and values
        # 计算动作和值
        self.transition.actions = self.actor_critic.act(obs).detach()     # 获取动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()  # 获取值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()  # 获取动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # 获取动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # 获取动作标准差
 
        # need to record obs and critic_obs before env.step()
        # 在环境步之前需要记录 obs 和 critic_obs
        self.transition.observations = obs                    # 记录观察
        self.transition.critic_observations = critic_obs      # 记录评论员观察
        return self.transition.actions          # 返回动作

    
    def process_env_step(self, rewards, dones, infos):  # 定义处理环境步函数
        self.transition.rewards = rewards.clone()       # 克隆奖励
        self.transition.dones = dones          # 记录完成状态
        # Bootstrapping on time outs
        # 对超时进行引导
        if 'time_outs' in infos:               # 如果信息中有 'time_outs'
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)  # 更新奖励
 
        # Record the transition
        # 记录过渡
        self.storage.add_transitions(self.transition)  # 添加过渡到存储中
        self.transition.clear()              # 清除过渡数据

    def compute_returns(self, last_critic_obs):        # 定义计算回报函数
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()  # 计算最后的值
        self.storage.compute_returns(last_values, self.gamma, self.lam)     # 计算回报


    # 根据模型是否是循环的来选择不同的mini-batch生成器。然后，它遍历生成器产生的每个mini-batch，对每个batch进行一系列的操作
    # 在每个mini-batch中，首先使用[`actor_critic`]模型对观察值进行行动，并获取行动的对数概率，评估价值，以及行动的均值和标准差
    def update(self):              # 定义更新函数
        mean_value_loss = 0        # 初始化平均值损失
        mean_surrogate_loss = 0    # 初始化平均代理损失
        if self.actor_critic.is_recurrent:  # 如果 actor_critic 是循环的
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)                   # 使用循环小批量生成器
        else:  # 否则
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)          # 使用小批量生成器
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:  # 迭代生成器中的数据
 
                # 获取动作概率和价值
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])                          # 获取动作
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)       # 获取动作的对数概率
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, 
                                                         hidden_states=hid_states_batch[1])           # 获取值
                
                mu_batch = self.actor_critic.action_mean        # 获取动作均值
                sigma_batch = self.actor_critic.action_std      # 获取动作标准差
                entropy_batch = self.actor_critic.entropy       # 获取熵

                # KL 
                # 如果设置了期望的KL散度并且调度策略为'adaptive'，则计算当前策略和旧策略之间的KL散度，并根据KL散度的大小动态调整学习率
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                # 代理损失
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))  # 计算比率
                surrogate = -torch.squeeze(advantages_batch) * ratio  # 计算代理损失
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)  # 计算剪切后的代理损失
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()  # 计算代理损失的均值

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                # 梯度步
                self.optimizer.zero_grad()  # 清零梯度
                loss.backward() # 反向传播
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)    #剪切梯度
                self.optimizer.step()   # 更新优化器

                mean_value_loss += value_loss.item()    # 累加平均值损失
                mean_surrogate_loss += surrogate_loss.item()    # 累加平均代理损失

        num_updates = self.num_learning_epochs * self.num_mini_batches  # 计算更新次数
        mean_value_loss /= num_updates  # 计算平均值损失
        mean_surrogate_loss /= num_updates  # 计算平均代理损失
        self.storage.clear()  # 清除存储
 
        return mean_value_loss, mean_surrogate_loss  # 返回平均值损失和平均代理损失