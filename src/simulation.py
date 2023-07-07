import bisect
import json
import logging
import operator
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cache, partial, reduce
from itertools import cycle, islice
from math import log2
from pathlib import Path
from random import choice
from statistics import mean
from typing import Callable, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO


class 事件类(object):
    def __init__(self, 时间: float, 回调函数: Callable[[], None]):
        object.__init__(self)
        self.时间 = 时间
        self.回调函数 = 回调函数


class 运行状态类(Enum):
    就绪 = auto()
    睡眠 = auto()
    忙碌 = auto()


class 离散事件仿真器类(object):
    事件队列: list[事件类] = []
    当前时间 = 0
    状态 = 运行状态类.就绪
    剩余运行时间 = 0

    @classmethod
    def 运行(cls, 时长: float) -> None:
        if cls.状态 != 运行状态类.就绪:
            raise RuntimeError()
        else:
            开始时间 = cls.当前时间
            结束时间 = 开始时间 + 时长
            while True:
                if cls.事件队列:
                    cls.状态 = 运行状态类.忙碌
                    if cls.事件队列[0].时间 >= 结束时间:
                        cls.状态 = 运行状态类.就绪
                        cls.当前时间 = 结束时间
                        break
                    当前事件 = cls.事件队列.pop(0)
                    cls.当前时间 = 当前事件.时间
                    当前事件.回调函数()
                else:
                    cls.状态 = 运行状态类.睡眠
                    cls.剩余运行时间 = 结束时间 - cls.当前时间
                    break

    @classmethod
    def 添加事件(cls, 时延: float, 回调函数: Callable[[], None]) -> None:
        bisect.insort(cls.事件队列, 事件类(cls.当前时间 + 时延, 回调函数), key=lambda 事件: 事件.时间)
        if cls.状态 == 运行状态类.睡眠:
            cls.状态 = 运行状态类.就绪
            cls.运行(cls.剩余运行时间)

    @classmethod
    def 重置(cls) -> None:
        cls.事件队列.clear()
        cls.剩余运行时间 = 0
        cls.当前时间 = 0
        cls.状态 = 运行状态类.就绪


class 编号类(object):
    实例数量 = 0

    def __init__(self):
        object.__init__(self)
        self.编号 = type(self).实例数量
        type(self).实例数量 += 1


class 流量类型类(Enum):
    时延敏感型 = auto()
    带宽敏感型 = auto()


class 数据流类(编号类):
    def __init__(self, 源节点: "节点类", 目的节点: "节点类", 类型: 流量类型类):
        编号类.__init__(self)
        self.源节点 = 源节点
        self.目的节点 = 目的节点
        self.流量类型 = 类型

    def 生成数据包(self, 数量: int, 字节数: int) -> list["数据包类"]:
        return [数据包类(self, 字节数) for _ in range(数量)]

    def __repr__(self):
        return f"<f{self.编号}>"


class 数据包类(编号类):
    def __init__(self, 数据流: 数据流类, 字节数: int):
        编号类.__init__(self)
        self.数据流 = 数据流
        self.字节数 = 字节数

    def __repr__(self):
        return f"<p{self.编号}: 3{'d' if self.数据流.流量类型 == 流量类型类.时延敏感型 else 'b'}>"


class 链路状态类(Enum):
    就绪 = auto()
    忙碌 = auto()
    关闭 = auto()


class 链路基类(ABC):
    def __init__(self, 源节点: "节点类", 目的节点: "节点类", 带宽: float | None = None):
        self.源节点 = 源节点
        self.目的节点 = 目的节点
        self.带宽 = 带宽
        self.数据包队列: list[数据包类] = []

    @abstractmethod
    def 发送(self, 数据包: 数据包类) -> None:
        pass


class 基本链路类(链路基类):
    def __init__(self, 源节点: "节点类", 目的节点: "节点类", 带宽: float | None = None):
        self._带宽 = 带宽
        self.状态 = 链路状态类.就绪
        链路基类.__init__(self, 源节点, 目的节点, 带宽)

    @property
    def 带宽(self) -> float | None:
        return self._带宽

    @带宽.setter
    def 带宽(self, 带宽_: float | None):
        self._带宽 = 带宽_
        if 带宽_ == 0:
            self.状态 = 链路状态类.关闭
        elif self.状态 == 链路状态类.关闭:
            self.状态 = 链路状态类.就绪
            self.开始传输()

    def 传输(self, 数据包: 数据包类) -> None:
        logging.info(f"{离散事件仿真器类.当前时间}s {self}开始传输{数据包}.")
        self.状态 = 链路状态类.忙碌
        传输时延 = 数据包.字节数 / self.带宽 if self.带宽 is not None else 0
        离散事件仿真器类.添加事件(传输时延, partial(self.目的节点.接收, 数据包))
        离散事件仿真器类.添加事件(传输时延, self.开始传输)

    def 开始传输(self) -> None:
        if self.状态 != 链路状态类.关闭:
            self.状态 = 链路状态类.就绪
            try:
                数据包 = self.数据包队列.pop(0)
            except IndexError:
                self.状态 = 链路状态类.就绪
            else:
                self.状态 = 链路状态类.忙碌
                self.传输(数据包)

    def 发送(self, 数据包: 数据包类) -> None:
        self.数据包队列.append(数据包)
        if self.状态 == 链路状态类.就绪:
            self.开始传输()

    def __repr__(self):
        return f"<logic link: {self.源节点} -> {self.目的节点}>"


class 独立带宽链路类(链路基类):
    def __init__(self,
                 源节点: "节点类",
                 目的节点: "节点类",
                 带宽: float | None = None,
                 带宽分配比例: dict[流量类型类, float] | None = None):
        self._数据包队列: list[数据包类] = []
        链路基类.__init__(self, 源节点, 目的节点, 带宽)
        self.基本链路表: dict[流量类型类, 基本链路类] = {流量类型: 基本链路类(源节点, 目的节点)
                                                         for 流量类型 in 流量类型类}
        self._带宽分配比例: dict[流量类型类, float] | None = None
        self._带宽分配: dict[流量类型类, float | None] | None = None
        self.带宽分配比例 = 带宽分配比例 if 带宽分配比例 is not None \
            else {流量类型: 1 / len(流量类型类) for 流量类型 in 流量类型类}

    @property
    def 数据包队列(self) -> list[数据包类]:
        return reduce(operator.add, [基本链路.数据包队列 for 基本链路 in self.基本链路表.values()])

    @数据包队列.setter
    def 数据包队列(self, 数据包队列_: list[数据包类]):
        pass

    @property
    def 带宽分配比例(self) -> dict[流量类型类, float]:
        return self._带宽分配比例

    @带宽分配比例.setter
    def 带宽分配比例(self, 带宽分配比例_: dict[流量类型类, float]):
        self._带宽分配比例 = 带宽分配比例_
        self.带宽分配 = {流量类型: self.带宽 * self._带宽分配比例[流量类型]
                         if self.带宽 is not None else None for 流量类型 in 流量类型类}

    @property
    def 带宽分配(self) -> dict[流量类型类, float | None]:
        return self._带宽分配

    @带宽分配.setter
    def 带宽分配(self, 带宽分配_: dict[流量类型类, float]):
        self._带宽分配 = 带宽分配_
        for 流量类型 in self.基本链路表:
            self.基本链路表[流量类型].带宽 = 带宽分配_[流量类型]

    def 发送(self, 数据包: 数据包类) -> None:
        self.基本链路表[数据包.数据流.流量类型].发送(数据包)

    def __repr__(self):
        return f"<link: {self.源节点} -> {self.目的节点}>"


class 节点类(编号类):
    def __init__(self):
        编号类.__init__(self)
        self.控制器: 控制器类 | None = None
        self.链路映射: dict[节点类, 链路基类] = {}
        self.流表: dict[数据流类, 节点类] = {}

    def __repr__(self):
        return f"<n{self.编号}>"

    def 发送(self, 数据包: 数据包类, 目的节点: "节点类") -> None:
        logging.info(f"{离散事件仿真器类.当前时间}s {self}发送数据包{数据包}到{目的节点}.")
        self.链路映射[目的节点].发送(数据包)

    def 接收(self, 数据包: 数据包类) -> None:
        logging.info(f"{离散事件仿真器类.当前时间}s {self}接收到{数据包}.")
        if 数据包.数据流.目的节点 is not self:
            try:
                下一跳节点 = self.流表[数据包.数据流]
            except KeyError:
                self.发送PacketIn消息(数据包)
                下一跳节点 = self.流表[数据包.数据流]
            self.发送(数据包, 下一跳节点)

    def 生成(self, 数据包: 数据包类) -> None:
        logging.info(f"{离散事件仿真器类.当前时间}s {self}生成数据包{数据包} {数据包.数据流.源节点} -> {数据包.数据流.目的节点}.")
        try:
            下一跳节点 = self.流表[数据包.数据流]
        except KeyError:
            self.发送PacketIn消息(数据包)
            下一跳节点 = self.流表[数据包.数据流]
        self.发送(数据包, 下一跳节点)

    def 发送PacketIn消息(self, 数据包: 数据包类) -> None:
        self.控制器.接收PacketIn消息(数据包)

    def 建立链路(self, 目的节点: "节点类", 链路类型: Type[基本链路类 | 独立带宽链路类], 带宽: float | None = None) -> None:
        self.链路映射[目的节点] = 链路类型(self, 目的节点, 带宽)


class 交换机类(节点类):
    def __init__(self):
        节点类.__init__(self)

    def __repr__(self):
        return f"<s{self.编号}>"


class 主机类(节点类):
    def __init__(self):
        节点类.__init__(self)
        self.发送数据包计数: dict[流量类型类, int] = {流量类型: 0 for 流量类型 in 流量类型类}
        self.接收数据包计数: dict[流量类型类, int] = {流量类型: 0 for 流量类型 in 流量类型类}
        self.发送数据包时间记录: dict[数据包类, float] = {}
        self.接收数据包时间记录: dict[数据包类, float] = {}

    def __repr__(self):
        return f"<h{self.编号}>"

    def 接收(self, 数据包: 数据包类) -> None:
        节点类.接收(self, 数据包)
        self.接收数据包计数[数据包.数据流.流量类型] += 1
        self.接收数据包时间记录[数据包] = 离散事件仿真器类.当前时间

    def 发送(self, 数据包: 数据包类, 目的节点: "节点类") -> None:
        节点类.发送(self, 数据包, 目的节点)
        self.发送数据包计数[数据包.数据流.流量类型] += 1
        self.发送数据包时间记录[数据包] = 离散事件仿真器类.当前时间


class 路由策略基类(ABC):
    def __init__(self):
        ABC.__init__(self)
        self._网络拓扑: nx.Graph | None = None

    @property
    def 网络拓扑(self) -> nx.Graph:
        return self._网络拓扑

    @网络拓扑.setter
    def 网络拓扑(self, 网络拓扑_: nx.Graph):
        self._网络拓扑 = 网络拓扑_

    @abstractmethod
    def 生成路由(self, 源节点: 节点类, 目的节点: 节点类) -> list[节点类]:
        pass


class 随机最短路由策略类(路由策略基类):
    def __init__(self):
        路由策略基类.__init__(self)
        self.路由表: dict[(主机类, 主机类), list[list[节点类]]] = {}

    @路由策略基类.网络拓扑.setter
    def 网络拓扑(self, 网络拓扑_: nx.Graph):
        self._网络拓扑 = 网络拓扑_
        self.路由表 = self.生成全局最短路径表(网络拓扑_)

    @staticmethod
    def 生成全局最短路径表(网络拓扑: nx.Graph) -> dict[(主机类, 主机类), list[list[节点类]]]:
        主机列表 = [节点 for 节点 in 网络拓扑.nodes if isinstance(节点, 主机类)]
        路由表 = {(源节点, 目的节点): list(nx.all_shortest_paths(网络拓扑, 源节点, 目的节点))
               for 源节点 in 主机列表 for 目的节点 in 主机列表 if 源节点 != 目的节点}
        return 路由表

    def 生成路由(self, 源节点: 节点类, 目的节点: 节点类) -> list[节点类]:
        最短路径列表 = self.路由表[(源节点, 目的节点)]
        路径 = choice(最短路径列表)
        return 路径


class 最小负载最短路由策略类(路由策略基类):
    def __init__(self):
        路由策略基类.__init__(self)
        self.路由表: dict[(主机类, 主机类), list[list[节点类]]] = {}

    @路由策略基类.网络拓扑.setter
    def 网络拓扑(self, 网络拓扑_: nx.Graph):
        self._网络拓扑 = nx.DiGraph()
        self._网络拓扑.add_nodes_from(网络拓扑_.nodes)
        for 边 in 网络拓扑_.edges:
            self._网络拓扑.add_edge(边[0], 边[1])
            self._网络拓扑.add_edge(边[1], 边[0])
            self._网络拓扑[边[0]][边[1]]["选择次数"] = 0
            self._网络拓扑[边[1]][边[0]]["选择次数"] = 0
        self.路由表 = self.生成全局最短路径表(网络拓扑_)

    @staticmethod
    def 生成全局最短路径表(网络拓扑: nx.Graph) -> dict[(主机类, 主机类), list[list[节点类]]]:
        主机列表 = [节点 for 节点 in 网络拓扑.nodes if isinstance(节点, 主机类)]
        路由表 = {(源节点, 目的节点): list(nx.all_shortest_paths(网络拓扑, 源节点, 目的节点))
               for 源节点 in 主机列表 for 目的节点 in 主机列表 if 源节点 != 目的节点}
        return 路由表

    def 生成路由(self, 源节点: 节点类, 目的节点: 节点类) -> list[节点类]:
        路径 = min(self.路由表[(源节点, 目的节点)], key=partial(nx.path_weight, self.网络拓扑, weight="选择次数"))
        for i in range(len(路径) - 1):
            self.网络拓扑[路径[i]][路径[i + 1]]["选择次数"] += 1
        return 路径


class 控制器类(object):
    def __init__(self, 路由策略类型: Type[路由策略基类]):
        object.__init__(self)
        self.路由策略 = 路由策略类型()

    @property
    def 网络拓扑(self) -> nx.Graph:
        return self.路由策略.网络拓扑

    @网络拓扑.setter
    def 网络拓扑(self, 网络拓扑_: nx.Graph):
        self.路由策略.网络拓扑 = 网络拓扑_

    def 接收PacketIn消息(self, 数据包: 数据包类) -> None:
        数据流 = 数据包.数据流
        路径 = self.路由策略.生成路由(数据流.源节点, 数据流.目的节点)
        self.下发流表(数据流, 路径)

    @staticmethod
    def 下发流表(数据流: 数据流类, 路径: list[节点类]) -> None:
        for i in range(len(路径) - 1):
            当前节点 = 路径[i]
            下一跳节点 = 路径[i + 1]
            当前节点.流表[数据流] = 下一跳节点


class 流量生成器类(object):
    def __init__(self,
                 主机列表: list[主机类],
                 流量生成速率: float,
                 每流数据包数: int,
                 数据包字节数: int,
                 流量类型: 流量类型类):
        object.__init__(self)
        self.主机列表 = 主机列表
        self.流量生成速率 = 流量生成速率
        self.每流数据包数 = 每流数据包数
        self.数据包字节数 = 数据包字节数
        self.流量类型 = 流量类型
        self.数据包生成间隔 = 1 / (self.流量生成速率 * self.每流数据包数)
        self.数据流生成间隔 = 1 / self.流量生成速率

    def 开始生成流量(self) -> None:
        for 源主机 in self.主机列表:
            其他主机列表 = list(islice(
                cycle(self.主机列表),
                self.主机列表.index(源主机) + 1,
                self.主机列表.index(源主机) + len(self.主机列表)
            ))
            for i in range(len(其他主机列表)):
                数据流 = 数据流类(源主机, choice(其他主机列表), self.流量类型)
                数据包列表 = 数据流.生成数据包(self.每流数据包数, self.数据包字节数)
                for j in range(self.每流数据包数):
                    离散事件仿真器类.添加事件(
                        i * self.数据流生成间隔 + j * self.数据包生成间隔,
                        partial(源主机.生成, 数据包列表[j])
                    )
        离散事件仿真器类.添加事件(self.数据流生成间隔 * (len(self.主机列表) - 1), self.开始生成流量)


def 建立双向链路(节点对: tuple[节点类, 节点类],
           链路类型: Type[独立带宽链路类 | 基本链路类],
           带宽: float | None = None) -> None:
    节点对[0].建立链路(节点对[1], 链路类型, 带宽)
    节点对[1].建立链路(节点对[0], 链路类型, 带宽)


def 生成FatTree(链路类型: Type[独立带宽链路类 | 基本链路类], 链路带宽: float, 控制器路由策略: Type[路由策略基类]) -> nx.Graph:
    控制器 = 控制器类(控制器路由策略)
    网络拓扑 = nx.Graph()
    核心层交换机列表 = [交换机类() for _ in range(4)]
    网络拓扑.add_nodes_from(核心层交换机列表)
    for i in range(4):
        汇聚层交换机列表 = [交换机类() for _ in range(2)]
        网络拓扑.add_nodes_from(汇聚层交换机列表)
        for j in range(2):
            接入层交换机 = 交换机类()
            网络拓扑.add_node(接入层交换机)
            for k in range(2):
                主机 = 主机类()
                网络拓扑.add_node(主机)
                网络拓扑.add_edge(主机, 接入层交换机)
                主机.建立链路(接入层交换机, 链路类型)
                接入层交换机.建立链路(主机, 链路类型, 链路带宽)
            for 汇聚层交换机 in 汇聚层交换机列表:
                网络拓扑.add_edge(接入层交换机, 汇聚层交换机)
                建立双向链路((接入层交换机, 汇聚层交换机), 链路类型, 链路带宽)
        网络拓扑.add_edge(汇聚层交换机列表[0], 核心层交换机列表[0])
        网络拓扑.add_edge(汇聚层交换机列表[0], 核心层交换机列表[1])
        网络拓扑.add_edge(汇聚层交换机列表[1], 核心层交换机列表[2])
        网络拓扑.add_edge(汇聚层交换机列表[1], 核心层交换机列表[3])
        建立双向链路((汇聚层交换机列表[0], 核心层交换机列表[0]), 链路类型, 链路带宽)
        建立双向链路((汇聚层交换机列表[0], 核心层交换机列表[1]), 链路类型, 链路带宽)
        建立双向链路((汇聚层交换机列表[1], 核心层交换机列表[2]), 链路类型, 链路带宽)
        建立双向链路((汇聚层交换机列表[1], 核心层交换机列表[3]), 链路类型, 链路带宽)
    控制器.网络拓扑 = 网络拓扑
    for 节点 in 网络拓扑.nodes:
        节点.控制器 = 控制器
    return 网络拓扑


def 基本方法仿真(路由策略类型: Type[路由策略基类],
                 链路总带宽: float,
                 流量生成速率: dict[流量类型类, float],
                 每流数据包数: dict[流量类型类, int],
                 数据包字节数: dict[流量类型类, int],
                 仿真时长: int) -> dict[str, float]:
    print("开始仿真:"
          f"路由策略类型: {路由策略类型}"
          f"链路总带宽: {链路总带宽}"
          f"流量生成速率: {流量生成速率}"
          f"每流数据包数: {每流数据包数}"
          f"数据包字节数: {数据包字节数}"
          f"仿真时长: {仿真时长}")
    离散事件仿真器类.重置()
    网络拓扑 = 生成FatTree(基本链路类, 链路总带宽, 路由策略类型)
    主机列表: list[主机类] = [节点 for 节点 in 网络拓扑.nodes if isinstance(节点, 主机类)]
    for 流量类型 in 流量类型类:
        流量生成器类(
            主机列表=主机列表,
            流量生成速率=流量生成速率[流量类型],
            每流数据包数=每流数据包数[流量类型],
            数据包字节数=数据包字节数[流量类型],
            流量类型=流量类型
        ).开始生成流量()
    离散事件仿真器类.运行(仿真时长)
    平均吞吐率 = sum([主机.接收数据包计数[流量类型] * 数据包字节数[流量类型] for 流量类型 in 流量类型类 for 主机 in 主机列表]) / 仿真时长
    数据包发送时间记录 = reduce(operator.or_, [主机.发送数据包时间记录 for 主机 in 主机列表])
    数据包接收时间记录 = reduce(operator.or_, [主机.接收数据包时间记录 for 主机 in 主机列表])
    时延敏感流量平均时延 = mean([数据包接收时间记录[数据包] - 数据包发送时间记录[数据包]
                                 for 数据包 in 数据包接收时间记录
                                 if 数据包.数据流.流量类型 == 流量类型类.时延敏感型])
    print(f"仿真完成. 平均吞吐率: {平均吞吐率}, 时延敏感流量平均时延: {时延敏感流量平均时延}")
    return {"平均吞吐率": 平均吞吐率, "时延敏感流量平均时延": 时延敏感流量平均时延}


class 强化学习环境类(Env):
    def __init__(self,
                 链路总带宽: float,
                 调整间隔: float,
                 时延阈值: float,
                 流量生成速率: dict[流量类型类, float],
                 每流数据包数: dict[流量类型类, int],
                 数据包字节数: dict[流量类型类, int]):
        Env.__init__(self)
        self.链路总带宽 = 链路总带宽
        self.调整间隔 = 调整间隔
        self.时延阈值 = 时延阈值
        self._网络拓扑 = 生成FatTree(独立带宽链路类, 链路总带宽, 随机最短路由策略类)
        self.流量生成速率 = 流量生成速率
        self.每流数据包数 = 每流数据包数
        self.数据包字节数 = 数据包字节数

        全局流量生成率 = sum([流量生成速率[流量类型] * 每流数据包数[流量类型] * 数据包字节数[流量类型] for 流量类型 in 流量类型类])
        全局链路总带宽 = self.链路总带宽 * len(self.主机列表) * 2
        最大可能吞吐率 = min([全局流量生成率, 全局链路总带宽])
        self.observation_space = Box(0, 最大可能吞吐率, (2,))
        self.action_space = Discrete(20)
        self.reward_range = (-float("inf"), self.action_space.n)

        self.是否训练 = True
        self.当前步数 = 0
        self.历史主机发送数据包计数: dict[流量类型类, int] = {流量类型: 0 for 流量类型 in 流量类型类}
        self.历史主机接收数据包计数: dict[流量类型类, int] = {流量类型: 0 for 流量类型 in 流量类型类}
        self.时延敏感流量累计时延 = 0

    @property
    @cache
    def 网络拓扑(self) -> nx.Graph:
        return self._网络拓扑

    @property
    @cache
    def 交换机列表(self) -> list[交换机类]:
        return [节点 for 节点 in self.网络拓扑 if isinstance(节点, 交换机类)]

    @property
    @cache
    def 主机列表(self) -> list[主机类]:
        return [节点 for 节点 in self.网络拓扑 if isinstance(节点, 主机类)]

    @property
    @cache
    def 逻辑链路列表(self) -> list[基本链路类]:
        return [链路.基本链路表[流量类型]
                for 交换机 in self.交换机列表
                for 链路 in 交换机.链路映射.values()
                for 流量类型 in 流量类型类]

    @property
    def 数据包列表(self) -> list[数据包类]:
        return [数据包 for 逻辑链路 in self.逻辑链路列表 for 数据包 in 逻辑链路.数据包队列]

    @property
    def 当前主机发送数据包计数(self) -> dict[流量类型类, int]:
        return {流量类型: sum([主机.发送数据包计数[流量类型] for 主机 in self.主机列表]) for 流量类型 in 流量类型类}

    @property
    def 当前主机接收数据包计数(self) -> dict[流量类型类, int]:
        return {流量类型: sum([主机.接收数据包计数[流量类型] for 主机 in self.主机列表]) for 流量类型 in 流量类型类}

    @property
    def 当前网络负载(self) -> np.ndarray:
        return np.array([(self.当前主机发送数据包计数[流量类型] - self.历史主机发送数据包计数[流量类型])
                         / self.调整间隔 for 流量类型 in 流量类型类], dtype=np.float32)

    @property
    def 当前网络吞吐率(self) -> float:
        return (sum(self.当前主机接收数据包计数.values())
                - sum(self.历史主机接收数据包计数.values())) \
               / self.调整间隔

    @property
    def 当前发送数据包时间记录(self) -> dict[数据包类, float]:
        return reduce(operator.or_, [主机.发送数据包时间记录 for 主机 in self.主机列表])

    @property
    def 当前接收数据包时间记录(self) -> dict[数据包类, float]:
        return reduce(operator.or_, [主机.接收数据包时间记录 for 主机 in self.主机列表])

    @property
    def 当前时延敏感流量平均时延(self) -> float:
        return mean([self.当前接收数据包时间记录[数据包] - self.当前发送数据包时间记录[数据包]
                     if 数据包 in self.当前接收数据包时间记录
                     else 离散事件仿真器类.当前时间 - self.当前发送数据包时间记录[数据包]
                     for 数据包 in self.当前发送数据包时间记录
                     if 数据包.数据流.流量类型 == 流量类型类.时延敏感型])

    @property
    def 当前间隔接收时延敏感流量累计时延(self) -> float:
        return sum([self.当前接收数据包时间记录[数据包] - self.当前发送数据包时间记录[数据包]
                    for 数据包 in self.当前接收数据包时间记录
                    if 数据包.数据流.流量类型 == 流量类型类.时延敏感型])

    def step(self, action: np.int64) -> (np.ndarray, float, bool, dict):
        if self.是否训练:
            self.重置()

        时延敏感流量带宽比例 = 2 ** (- 1 - int(action))
        带宽分配比例 = {流量类型类.时延敏感型: 时延敏感流量带宽比例, 流量类型类.带宽敏感型: 1 - 时延敏感流量带宽比例}
        self.设置全局带宽分配比例(带宽分配比例)

        离散事件仿真器类.运行(self.调整间隔)
        self.当前步数 += 1

        观察值 = self.当前网络负载
        奖励 = -log2(带宽分配比例[流量类型类.时延敏感型]) - self.当前时延敏感流量平均时延 \
               if self.当前时延敏感流量平均时延 < self.时延阈值 \
               and 带宽分配比例[流量类型类.带宽敏感型] > 0 \
               else 1 - self.当前时延敏感流量平均时延
        print(f"步数: {self.当前步数}, "
              f"动作: {时延敏感流量带宽比例}, "
              f"观察值: {观察值}, "
              f"奖励: {奖励}, "
              f"平均排队时延: {self.当前时延敏感流量平均时延}, "
              f"当前网络吞吐率: {self.当前网络吞吐率}")

        self.历史主机发送数据包计数 = self.当前主机发送数据包计数
        self.历史主机接收数据包计数 = self.当前主机接收数据包计数
        self.时延敏感流量累计时延 += self.当前间隔接收时延敏感流量累计时延
        self.清理数据包时间记录()
        self.清理流表()

        return 观察值, 奖励, False, {}

    def reset(self) -> np.ndarray:
        self.重置()
        self.当前步数 = 0
        return self.当前网络负载

    def render(self, mode="human"):
        pass

    def 重置(self) -> None:
        离散事件仿真器类.重置()
        for 流量类型 in 流量类型类:
            流量生成器类(
                主机列表=self.主机列表,
                流量生成速率=self.流量生成速率[流量类型],
                每流数据包数=self.每流数据包数[流量类型],
                数据包字节数=self.数据包字节数[流量类型],
                流量类型=流量类型
            ).开始生成流量()
        for 主机 in self.主机列表:
            主机.发送数据包计数 = {流量类型: 0 for 流量类型 in 流量类型类}
            主机.接收数据包计数 = {流量类型: 0 for 流量类型 in 流量类型类}
            主机.接收数据包时间记录.clear()
            主机.发送数据包时间记录.clear()
            主机.流表.clear()
        for 交换机 in self.交换机列表:
            交换机.流表.clear()
        for 逻辑链路 in self.逻辑链路列表:
            逻辑链路.数据包队列.clear()
            逻辑链路.状态 = 链路状态类.就绪
        self.历史主机发送数据包计数 = {流量类型: 0 for 流量类型 in 流量类型类}
        self.历史主机接收数据包计数 = {流量类型: 0 for 流量类型 in 流量类型类}
        self.时延敏感流量累计时延 = 0

    def 设置全局带宽分配比例(self, 带宽分配比例: dict[流量类型类, float]) -> None:
        for 交换机 in self.交换机列表:
            for 链路 in 交换机.链路映射.values():
                链路.带宽分配比例 = 带宽分配比例

    def 清理流表(self) -> None:
        for 节点 in self.网络拓扑.nodes:
            节点.流表 = {数据包.数据流: 节点.流表[数据包.数据流]
                     for 数据包 in self.当前发送数据包时间记录
                     if 数据包.数据流 in 节点.流表}

    def 清理数据包时间记录(self) -> None:
        for 主机 in self.主机列表:
            while True:
                try:
                    数据包 = 主机.接收数据包时间记录.popitem()[0]
                except KeyError:
                    break
                else:
                    源主机: 主机类 = 数据包.数据流.源节点
                    源主机.发送数据包时间记录.pop(数据包)

    def 清空数据包时间记录(self) -> None:
        for 主机 in self.主机列表:
            主机.接收数据包时间记录.clear()
            主机.发送数据包时间记录.clear()


def 强化学习方法仿真(链路总带宽: float,
                     流量生成速率: dict[流量类型类, float],
                     每流数据包数: dict[流量类型类, int],
                     数据包字节数: dict[流量类型类, int],
                     仿真时长: int) -> dict[str, float]:
    环境 = 强化学习环境类(
        链路总带宽=链路总带宽,
        调整间隔=1,
        时延阈值=1e-1,
        流量生成速率=流量生成速率,
        每流数据包数=每流数据包数,
        数据包字节数=数据包字节数
    )
    模型 = PPO(
        policy="MlpPolicy",
        env=环境,
        learning_rate=0.003,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0
    )
    模型.learn(100)

    观察值 = 环境.reset()
    环境.是否训练 = False
    步数 = round(仿真时长 / 环境.调整间隔)
    for i in range(步数):
        动作, _ = 模型.predict(观察值, deterministic=True)
        观察值, 奖励, _, _ = 环境.step(int(动作))

    平均吞吐率 = sum([环境.历史主机接收数据包计数[流量类型] * 数据包字节数[流量类型] for 流量类型 in 流量类型类]) / 仿真时长
    时延敏感流量平均时延 = 环境.时延敏感流量累计时延 / 环境.历史主机接收数据包计数[流量类型类.时延敏感型]
    return {"平均吞吐率": 平均吞吐率, "时延敏感流量平均时延": 时延敏感流量平均时延}


class 路由策略类型类(Enum):
    随机最短路由策略 = 随机最短路由策略类
    最小负载路由策略 = 最小负载最短路由策略类
    强化学习带宽分配策略 = auto()


def 仿真(路由策略类型: 路由策略类型类,
         链路总带宽: float,
         流量生成速率: dict[流量类型类, float],
         每流数据包数: dict[流量类型类, int],
         数据包字节数: dict[流量类型类, int],
         仿真时长: int) -> dict[str, float]:
    if 路由策略类型.value in 路由策略基类.__subclasses__():
        return 基本方法仿真(
            路由策略类型.value,
            链路总带宽,
            流量生成速率,
            每流数据包数,
            数据包字节数,
            仿真时长
        )
    elif 路由策略类型 == 路由策略类型类.强化学习带宽分配策略:
        return 强化学习方法仿真(
            链路总带宽,
            流量生成速率,
            每流数据包数,
            数据包字节数,
            仿真时长
        )


def 仿真画图(保存路径: Path, 仿真结果: dict[str, dict[str, dict[float, dict[str, float]]]] | None = None) -> None:
    if 仿真结果 is None:
        仿真结果 = json.loads(Path(保存路径 / "仿真结果.json").read_text("utf-8"))
    mpl.rc("font", family="SimSun")
    宽度 = 0.2
    性能指标单位 = {"平均吞吐率": "b/s", "时延敏感流量平均时延": "s"}
    可变参数单位 = {"主机平均流量大小": "b/s", "时延敏感流量比例": "倍"}
    for 性能指标 in ["平均吞吐率", "时延敏感流量平均时延"]:
        for 可变参数 in 仿真结果:
            仿真图, 坐标轴 = plt.subplots(figsize=[5.6, 3.6], dpi=300)
            仿真图: plt.Figure
            坐标轴: plt.Axes
            方法列表 = list(仿真结果[可变参数].keys())
            横坐标 = np.arange(len(仿真结果[可变参数][方法列表[0]]))
            for i in range(len(方法列表)):
                性能指标值列表 = [仿真结果[可变参数][方法列表[i]][可变参数值][性能指标]
                                  for 可变参数值 in 仿真结果[可变参数][方法列表[i]]]
                坐标轴.bar(
                    横坐标 + 宽度 * (i + 0.5 * (1 - len(方法列表))),
                    性能指标值列表,
                    宽度,
                    label=方法列表[i],
                    linewidth=0.5,
                    edgecolor="black"
                )
            坐标轴.set_ylabel(f"{性能指标} ({性能指标单位[性能指标]})")
            坐标轴.set_xlabel(f"{可变参数} ({可变参数单位[可变参数]})")
            坐标轴.set_xticks(横坐标, list(map(lambda x: np.format_float_positional(float(x), trim="-"), 仿真结果[可变参数][方法列表[0]])))
            坐标轴.legend(ncols=len(方法列表), fontsize="small", loc="upper center", bbox_to_anchor=(0.5, 1.15))
            仿真图.tight_layout()
            仿真图.savefig(str(保存路径 / f"{可变参数}{性能指标}对比图.png"))


if __name__ == "__main__":
    # logging.basicConfig(filename="../数据/info.txt", filemode='w', level=logging.INFO)
    链路总带宽 = 1.5e4
    基本总流量大小 = 1e4
    总流量大小列表 = [[0.7, 0.8, 0.9, 1][i] * 基本总流量大小 for i in range(4)]
    基本时延敏感流量比例 = 1e-2
    时延敏感流量比例列表 = [10 ** (-i) for i in range(2, 6)]
    基本流量比例 = {流量类型类.时延敏感型: 基本时延敏感流量比例, 流量类型类.带宽敏感型: 1 - 基本时延敏感流量比例}
    流量比例列表 = [{
        流量类型类.时延敏感型: 时延敏感流量比例,
        流量类型类.带宽敏感型: 1 - 时延敏感流量比例
    } for 时延敏感流量比例 in 时延敏感流量比例列表]
    流量生成速率 = {流量类型类.时延敏感型: 15, 流量类型类.带宽敏感型: 15}
    每流数据包数 = {流量类型类.时延敏感型: 1, 流量类型类.带宽敏感型: 1}
    数据包字节数 = {流量类型类.时延敏感型: int(1e3), 流量类型类.带宽敏感型: int(1e3)}
    仿真时长 = 10

    仿真结果 = {
        "流量大小": {
            路由策略类型.name: {
                流量大小: 仿真(
                    路由策略类型,
                    链路总带宽,
                    流量生成速率,
                    每流数据包数,
                    {流量类型: 流量大小 * 基本流量比例[流量类型] / (每流数据包数[流量类型] * 流量生成速率[流量类型])
                     for 流量类型 in 流量类型类},
                    仿真时长
                ) for 流量大小 in 总流量大小列表
            } for 路由策略类型 in 路由策略类型类
        },
        "时延敏感流量比例": {
            路由策略类型.name: {
                时延敏感流量比例列表[i]: 仿真(
                    路由策略类型,
                    链路总带宽,
                    流量生成速率,
                    每流数据包数,
                    {流量类型: 基本总流量大小 * 流量比例列表[i][流量类型] / (每流数据包数[流量类型] * 流量生成速率[流量类型])
                     for 流量类型 in 流量类型类},
                    仿真时长
                ) for i in range(len(流量比例列表))
            } for 路由策略类型 in 路由策略类型类
        }
    }
    仿真结果目录 = Path('../数据/仿真结果')
    仿真序号 = max(map(int, os.listdir(仿真结果目录))) + 1 if os.listdir(仿真结果目录) else 0
    仿真结果路径 = 仿真结果目录 / f"{仿真序号}"
    仿真结果路径.mkdir(parents=True)
    (仿真结果路径 / "仿真结果.json").write_text(json.dumps(仿真结果, ensure_ascii=False, indent=4), "utf-8")
    仿真画图(仿真结果路径, 仿真结果)
