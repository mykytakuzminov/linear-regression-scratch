# Linear Regression (From Scratch)

[![CI Status](https://github.com/mykytakuzminov/linear-regression-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/mykytakuzminov/linear-regression-scratch/actions)
![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)
![Tests: Pytest](https://img.shields.io/badge/tests-pytest-white.svg?logo=pytest&logoColor=white&labelColor=0a9edc)
![Types: MyPy](https://img.shields.io/badge/types-mypy-blue.svg?labelColor=2f4f4f)
![Linting: Ruff](https://img.shields.io/badge/linting-ruff-000000.svg?logo=python&logoColor=white)
![Automation: Tox](https://img.shields.io/badge/automation-tox-white.svg?logo=python&logoColor=white&labelColor=ce3262)

## 📝 Overview

This project is an educational implementation of **Linear Regression from scratch**, built using pure Python. The goal is to demystify the mechanics of machine learning by constructing the fundamental building blocks—specifically a custom matrix engine—without relying on heavy frameworks like NumPy or PyTorch.

This repository serves as a sandbox for understanding:
* **Linear Algebra**: Manual implementation of matrix operations (dot products, transpositions, etc.).
* **Optimization**: How Gradient Descent iteratively minimizes the cost function.
* **Loss Functions**: Implementing Mean Squared Error.

## 🚀 Engineering Stack

* **[Python 3.14](https://www.python.org/)**: Core logic implementation.
* **[Pytest](https://docs.pytest.org/)**: Unit testing to verify matrix operations and model convergence.
* **[MyPy](http://mypy-lang.org/)**: Strict static type checking to ensure architectural integrity.
* **[Ruff](https://github.com/astral-sh/ruff)**: Ultra-fast linter and formatter for PEP 8 compliance.
* **[Tox](https://tox.wiki/)**: Multi-environment orchestration for consistent testing.
* **[GitHub Actions](https://github.com/features/actions)**: Automated CI pipeline validating every commit.

## 🛠 Core Components

### 1. Matrix Engine
A custom `Matrix` class handling 2D numerical operations from scratch, including `__add__`, `__sub__`, `__mul__` (dot product), and `transpose`. This avoids dependency on libraries like NumPy.

### 2. Linear Regression Model
The model uses the fundamental linear equation:
$$y = Xw + b$$
Where:
* $X$ is the input feature matrix.
* $w$ represents the weights.
* $b$ is the bias.

The `fit` method implements **Batch Gradient Descent** to iteratively update parameters by minimizing the loss function.

### 3. Loss Function
We use the Mean Squared Error (MSE) to quantify the model's performance during training.

## ⚙️ Development Setup

Follow these steps to set up the project locally for development and testing. These instructions cover **macOS**, **Linux**, and **Windows**.

### 1. Clone

Start by cloning the repository to your local machine and navigating into the project directory.

```bash
git clone https://github.com/mykytakuzminov/linear-regression-scratch.git
cd linear-regression-scratch
```

### 2. Environment Setup

It is highly recommended to use a virtual environment to keep dependencies isolated. Choose the commands that match your operating system and shell.

#### Windows (PowerShell)

**Create a virtual environment**

```powershell
python -m venv .venv
```

**Activate the environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

**Create a virtual environment**

```bash
python3 -m venv .venv
```

**Activate the environment**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

This project follows modern Python packaging standards. Installing the package in editable mode ensures that any changes you make to the source code are instantly available without needing to reinstall.

**Upgrade pip to the latest version**

```bash
pip install --upgrade pip
```

**Install the package in editable mode with development dependencies**

```bash
pip install -e .
```

### 6. Running Tests & Quality Control

To maintain high code quality, this project uses **Tox** to automate testing and linting in isolated environments.

**Run everything at once**

```bash
tox
```

**Run only unit tests**

```bash
tox -e py314
```

**Run only linter (Ruff)**

```bash
tox -e ruff
```

**Run only type checking (MyPy)**

```bash
tox -e mypy
```
