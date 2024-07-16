
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """伯努利老虎机，k表示拉杆的数量"""
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)  # 随机生成k个0~1的数，作为拉动每根拉杆的获得奖励的概率
        self.best_idx = np.argmax(self.probs)  # 获得奖励概率最大的拉杆,argmax返回最大值的索引
        self.best_prob = self.probs[self.best_idx]  # 获奖的最大概率
        self.k = k

    def step(self, k):
        # 当玩家选择了k好拉杆后，根据拉动该老虎机的K号拉杆获得奖励的概率返回1
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

    def print_prob(self):
        for i in range(self.k):
            print("%d号拉杆的获奖概率为:%.4f" % (i, self.probs[i]))


np.random.seed(1) # 设定随机种子，使实验具有可重复性
k=10
bandit_10_arm=BernoulliBandit(k)

print("随机生成了一个%d臂伯努利老虎机"%k)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f"%(bandit_10_arm.best_idx,bandit_10_arm.best_prob))

print("************")
bandit_10_arm.print_prob()


class Solver:
    """多臂老虎机算法基础框架"""
    def __init__(self, bandit):
        self.bandit = bandit  # 多臂老虎机
        self.counts = np.zeros(self.bandit.k)  # 计数器,初始化为0，记录每根拉杆被拉动的次数
        self.regret = 0  # 当前的累计懊悔
        self.actions = []  # 记录每一步的动作，即选择的拉杆的编号
        self.regrets = []  # 记录每一步的累积懊悔，即每一步的懊悔值
    def update_regret(self, k): # 累计懊悔是根据每次选择的拉杆的概率与最大获得奖励的概率之差来计算的
        # 计算累积懊悔并保存，k为本次选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]  # 选择的拉杆的概率与最大获得奖励的概率之差，懊悔
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError # NotImplementedError是一个异常类，用于指示一个类的方法尚未实现,需要子类重写,否则会报错

    def run(self, num_steps):
        # 运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()  # 决策结果
            self.counts[k] += 1 # 选择的拉杆被拉动次数加1
            self.actions.append(k)
            self.update_regret(k) # 更新累积懊悔


# 𝜖 =0.01,T=5000
class EpsionGreedy(Solver):
    """epsilon贪婪算法，继承自Solver类"""

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0): # 初始奖励给的很高，表示初始对每根拉杆都有机会选择
        super(EpsionGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.k)  # 初始化拉动所有拉杆的期望奖励估值

    def run_one_step(self):  # 父类Solver的方法重写
        if np.random.random() < self.epsilon: # 以epsilon的概率随机选择一根拉杆
            k = np.random.randint(0, self.bandit.k)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        r = self.bandit.step(k)  # 获得本次动作的奖励

        # 更新期望奖励估值,方法是每次增量都是1/(counts[k]+1)*(r-estimates[k])
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 增量式更新期望

        return k


# 可视化
def plot_results(solvers, solver_names):
    """生成累计懊悔随时间变化的图像。输入solvers是一个列表，列表中的每个元素是一种特定的策略
       而solver_names也是一个列表，存储每个策略的名称
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.k)
    plt.legend()
    plt.show()


np.random.seed(0)
epsilons=[1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver_list=[EpsionGreedy(bandit_10_arm,epsilon=e) for e in epsilons]

epsilon_greedy_solver_names=["epsilon={}".format(e) for e in epsilons]

for solver in epsilon_greedy_solver_list:
    solver.run(5000) # 每个策略运行5000次

plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)


class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的𝜖-贪婪算法，继承Solver类"""

    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.total_count = 0  # 执行次数，作为衰减因子

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count: # epsilon随时间衰减 epsilon=1/衰减因子
            k = np.random.randint(self.bandit.k)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)

        self.estimates[k] += 1 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k

np.random.seed(1)
decaying_epsilon_greedy_solver=DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print("𝜖衰减的贪婪算法的累计懊悔为：",decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver],['DecayingEpsilonGreedy'])


class UCB(Solver):
    """UCB算法，继承自Solver类"""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算置信上界

        k = np.argmax(ucb)
        r = self.bandit.step(k)

        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(1)
coef=1 # 不确定性系数
UCB_solver=UCB(bandit_10_arm,coef)
UCB_solver.run(5000)

print("上置信界算法的累计懊悔为：",UCB_solver.regret)
plot_results([UCB_solver],["UCB"])


class ThompsonSampling(Solver):
    """汤姆森采样算法，继承Solver类"""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.a = np.ones(self.bandit.k)  # 每根拉杆奖励为1的次数
        self.b = np.ones(self.bandit.k)  # 每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self.a, self.b) # 从beta分布中采样 a,b是beta分布的参数
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self.a[k] += r
        self.b[k] += (1 - r)

        return k

np.random.seed(1)
thompson_sampling_solver=ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print("汤普森采样的累计懊悔为：",thompson_sampling_solver.regret)

plot_results([thompson_sampling_solver],["ThompsonSampling"])

