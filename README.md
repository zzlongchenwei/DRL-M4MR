# DRL-M4MR: An intelligent multicast routing approach based on DQN deep reinforcement learning in SDN

Traditional multicast routing methods have some problems in constructing a multicast tree. These problems include limited access to network state information, poor adaptability to dynamic and complex changes in the network, and inflexible data forwarding. To address these defects, the optimal multicast routing problem in software-defined networking (SDN) is tailored as a multiobjective optimization problem, and DRL-M4MR, an intelligent multicast routing algorithm based on the deep Q network (DQN) deep reinforcement learning (DRL) method is designed to construct a multicast tree in a software-defined network. First, combining the characteristics of SDN global network-aware information, the multicast tree state matrix, link bandwidth matrix, link delay matrix and link packet loss rate matrix are designed as the state space of the reinforcement learning agent to solve the problem in that the original method cannot make full use of network status information. Second, the action space of the agent is all the links in the network, and the action selection strategy is designed to add the links to the current multicast tree in four cases. Third, single-step and final reward function forms are designed to guide the agent to make decisions to construct the optimal multicast tree. The double network architectures, dueling network architectures and prioritized experience replay are adopted to improve the learning efficiency and convergence of the agent. Finally, after the DRL-M4MR agent is trained, the SDN controller installs the multicast flow entries by reversely traversing the multicast tree to the SDN switches to implement intelligent multicast routing. The experimental results show that, compared with existing algorithms, the multicast tree constructed by DRL-M4MR can obtain better bandwidth, delay, and packet loss rate performance after training, and it can make more intelligent multicast routing decisions in a dynamic network environment.


# DRL-M4MR

```
.
├─mininet  
│  ├─iperfTM  # iperf发流的shell脚本文件
│  ├─links_info  # 链路最大带宽信息
│  ├─tm_statistic  # 生成流量矩阵的中间npy文件
│  └─topologies  # 网络拓扑信息
├─RL
│  │  config.py  # 配置文件
│  │  env.py  # 强化学习中的环境，负责动作执行后状态如何改变，输出下一个状态，或者不合法的动作的判断等
│  │  log.py  # 日志脚本
│  │  net.py  # 神经网络 nn.Module
│  │  replymemory.py  # 经验回放
│  │  rl.py  # 强化学习，如何更新Q值，动作选择，神经网络参数更新
│  │  train.py  # 其实就是main，调用上面的各个py，然后画图啥的
│  ├─data  # 训练时候的数据信息如bw、delay、loss、reward等的npy文件，方便后续处理
│  ├─images  # 结果的图
│  ├─Logs  # 日志
│  ├─runs  # tensorboard这个库自己创的存一些tensorboard网页显示数据的文件
│  ├─saved_agents  # 神经网络的权重文件
└─ryu
    ├─pickle  # 采集的整个网络的链路信息的pickle文件，方便处理
    └─weight  # 采集的整个网络的链路信息的csv文件，方便查看
```

运行逻辑：Mininet和Ryu配合，首先根据生成的流量矩阵，通过iperf发流，模拟变化环境，然后Ryu采集变化环境的链路状态信息（剩余带宽，时延和丢包率）并保存为pkl。强化学习根据这些pkl进行训练（因为是离线训练，所以这部分不需要Ryu和Mininet参与）并保存神经网络权重。训练完成后，在Ryu中只需加载神经网络权重，然后进行决策。

因为之前ubuntu系统电脑的硬盘损坏的缘故，有部分部署的代码丢失了。你可以将部分函数注释，然后根据上面思路实现就行。重点在强化学习部分。

## 修改哪些配置才能运行

```python
# 先到RL目录下，然后，点开config.py
# 如果你是windows系统，修改83行和86行，只用把前面的路径改成你的路径就行，如
r'D:\WorkSpace\Hello_Myself\Hello_Multicast\RLMulticastProject\mininet\topologies\topology2.xml'
r"D:\WorkSpace\Hello_Myself\Hello_Multicast\RLMulticastProject\ryu\pickle\2022-03-11-19-40-21"
# 改为
r'[你的路径]\DRL_M4MR\mininet\topologies\topology2.xml'
r"[你的路径]\DRL_M4MR\ryu\pickle\2022-03-11-19-40-21"
```



## 如何运行

```shell
# 先到RL目录下，然后
python train.py --mode [train, test, testp, testc]  
# 你可以选里面的一个，如
python trian.py --mode test
# 具体请看RL/trian.py 976行，改 default 也行
```



## 修改训练pkl个数，也就是时隙个数，也就是论文中的NLIs个数

```python
# 先到RL目录下，然后，点开config.py
# 第28行 PKL_NUM 这个常量，还有第27行 每隔多少个PKL取一个，也就是间隔 PKL_STEP
```



## 神经网络权重在哪

```python
# 神经网络权重保存在 RL/saved_agents 中
# 在 RL/train.py 第979行修改 default 就行
```



## 路径在哪个属性中

```python
# 组播路径在 env.py 的 MulticastEnv.route_graph 中保存，是一个networkx这个库的Graph()实例，
# 如 MulticastEnv这个类，在Train这个类中被实例化为self.env，那么就可以通过self.env.route_graph调用这个属性，然后进行networkx里的一些操作，如取这个图对象的边，就用

print(list(self.env.route_graph.edges))

# 更多操作可以在 https://networkx.org/documentation/stable/reference/index.html 查看
```

## train.py中的主要函数

```python
# 415行的 update()，其他函数多数为画图的
# compare_test()的逻辑和update()差不多，就是加载了训练好的权重，然后agent动作选择时每次都选Q值最大的就行
```

# Mininet脚本中的一些

```shell
# 运行 generate_matrices.py 生成流量矩阵， 命令行或者直接pycharm里点运行
python generate_matrices.py
# 运行 iperf_script.py 将流量矩阵转化成发流的shell脚本
python iperf_script.py
# 运行mininet的话就
python generate_nodes_topo.py
```

