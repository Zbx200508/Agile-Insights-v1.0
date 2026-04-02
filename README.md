# 敏捷洞察1.0

一个面向培训资料与行业白皮书的结构化总结助手。

## 当前版本能力

- 支持 PDF 上传
- 自动保存原文件
- 自动解析 PDF 文本
- 自动生成一页摘要
- 自动生成三级逻辑大纲
- 支持基于原文的问答
- 返回引用片段，提升回答可追溯性

## 当前项目形态

这是一个本地 MVP 版本，主要用于验证完整产品闭环。

当前技术实现：
- 后端：FastAPI
- PDF 解析：PyMuPDF
- 模型调用：OpenAI 兼容接口
- 数据存储：本地文件系统

## 项目目录结构

Agile-Insights-v1.0/
├─ app/
│  ├─ main.py
│  ├─ routes/
│  ├─ services/
│  ├─ static/
│  ├─ templates/
│  └─ __init__.py
├─ data/
│  ├─ uploads/
│  ├─ parsed/
│  └─ outputs/
├─ docs/
├─ tests/
├─ .env
├─ .env.example
├─ .gitignore
├─ README.md
└─ requirements.txt

## 本地运行方式

### 1. 创建虚拟环境
python -m venv .venv

### 2.安装依赖
.venv\Scripts\python.exe -m pip install -r requirements.txt

### 3.配置环境变量
API_KEY=真实key
API_BASE=模型服务地址
MODEL_NAME=模型名

### 4.启动服务
.venv\Scripts\python.exe -m uvicorn app.main:app --reload