# AI-Robot: SAM3 + GR-ConvNet 文字驱动抓取系统

基于 **SAM3（文字分割）** 和 **GR-ConvNet（抓取姿态估计）** 的机械臂智能抓取系统。

## 功能

1. 在画面底部输入物体英文名称
2. SAM3 自动从相机画面中分割出目标物体
3. GR-ConvNet 在物体区域计算最佳抓取姿态（角度、宽度）
4. xArm 机械臂自动执行抓取和放置

## 项目结构

```
AI-Robot/
├── main.py                  # 主入口
├── run.bat                  # 一键启动（Windows）
├── requirements.txt         # 依赖列表
├── README.md
├── models/
│   └── grconvnet3_cornell_rgbd   # GR-ConvNet 预训练权重
├── camera/
│   ├── rs_camera.py         # RealSense D435 相机驱动
│   └── utils.py             # 图像拼接工具
├── grasp/
│   ├── grconvnet_torch.py   # GR-ConvNet 推理封装
│   ├── robot_grasp.py       # 机械臂抓取控制
│   └── helpers/
│       ├── matrix_funcs.py  # 坐标变换（欧拉角 <-> 旋转矩阵）
│       └── covariance.py    # 协方差生成
└── inference/
    └── models/
        ├── grasp_model.py   # 网络基类（模型反序列化需要）
        └── grconvnet3.py    # GR-ConvNet3 网络定义
```

## 环境要求

- **Python 环境**: `sam3` conda 环境（`D:/miniconda3/envs/sam3/python.exe`）
- **CUDA**: 需要 NVIDIA GPU + CUDA（SAM3 推理）
- **硬件**:
  - Intel RealSense D435 深度相机
  - UFACTORY xArm 机械臂（默认 IP: `192.168.1.236`）

## 安装

```bash
# 1. 激活 sam3 环境
conda activate sam3

# 2. 安装依赖
pip install -r requirements.txt

# 3. 确保 SAM3 已安装（见 SAM3 项目文档）
# 4. 确保 models/grconvnet3_cornell_rgbd 权重文件存在
```

## 运行

```bash
# 方式1: 直接运行
D:/miniconda3/envs/sam3/python.exe main.py

# 方式2: 双击 run.bat
```

## 快捷键

| 按键 | 功能 |
|------|------|
| Enter | 确认输入的物体名称 |
| R | 重置（清除当前目标） |
| Esc | 退出程序 |

## 配置

在 `main.py` 顶部可修改以下参数：

- `ROBOT_IP` — 机械臂 IP 地址
- `SAM3_ROOT` — SAM3 项目路径
- `EULER_EEF_TO_COLOR_OPT` — 手眼标定参数
- `GRASPING_RANGE` — 抓取工作区域范围
- `DETECT_XYZ` — 检测位置
- `RELEASE_XYZ` — 释放位置
