# 模型预训练

## VsCode 项目启动

* 环境搭建

```shell
cd ./

# 创建虚拟环境
py -m venv .venv
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖 部分可能需要科学上网
pip install -r requirements.txt

# cuda 环境

# 这东西网络有点差
pip install torch  --index-url https://download.pytorch.org/whl/cu126

# torch 官网
https://pytorch.org/get-started/locally/


# 安装代码风格检查 autopep8
py -m pip install -U autopep8
```

* 项目启动

```shell
# 确保激活文件夹
.venv\Scripts\activate

# 启动项目
py main.py
```

* 训练脚本

```shell
# 数据清洗
py src/format.py

# 分词器格式化
py src/tokenizer.py

# 原始文本可能有版权之争，需要自行处理
```
