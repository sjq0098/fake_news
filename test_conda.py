#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
import importlib.util

def check_python_version(major_required=3, minor_required=7):
    """
    检查 Python 版本是否满足要求
    """
    if sys.version_info < (major_required, minor_required):
        raise RuntimeError(
            f"当前 Python 版本为 {sys.version_info.major}.{sys.version_info.minor}，"
            f"需要 >= {major_required}.{minor_required} 才能运行。"
        )
    else:
        print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} 版本满足要求！")

def check_package_installed(package_name, version_check=None):
    """
    检查某个依赖包是否安装，如需检查版本，可在 version_check 参数中指定 (>=, <=, ==, 等)
    """
    try:
        pkg_spec = importlib.util.find_spec(package_name)
        if pkg_spec is None:
            raise ImportError
        module = importlib.import_module(package_name)
        if version_check is not None:
            # 举例，要求 package_name 的版本 >= 2.0.0
            # version_check: ("ge", "2.0.0") 表示版本 >= 2.0.0
            compare_op, required_version_str = version_check
            current_version = getattr(module, "__version__", None)

            if current_version is None:
                print(f"[WARN] 无法检测 {package_name} 的版本，跳过版本比较...")
            else:
                from packaging import version
                current_v = version.parse(current_version)
                required_v = version.parse(required_version_str)

                if compare_op == "ge" and not (current_v >= required_v):
                    raise RuntimeError(f"{package_name} 版本过低: {current_version}, 需要 >= {required_version_str}")
                elif compare_op == "le" and not (current_v <= required_v):
                    raise RuntimeError(f"{package_name} 版本过高: {current_version}, 需要 <= {required_version_str}")
                elif compare_op == "eq" and not (current_v == required_v):
                    raise RuntimeError(f"{package_name} 版本不匹配: {current_version}, 需要 == {required_version_str}")

        print(f"[OK] {package_name} 已安装！")
    except ImportError:
        raise ImportError(f"未检测到 {package_name}，请先安装该包！")

def check_gpu_with_pytorch():
    """
    如果需要使用 PyTorch 并检查 GPU，可用性
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("[OK] PyTorch 检测到 GPU 可用！")
        else:
            print("[WARN] PyTorch 未检测到 GPU，若需要 GPU 加速，请检查 CUDA 环境。")
    except ImportError:
        print("[WARN] 未检测到 PyTorch，请安装后再做 GPU 检测。")


if __name__ == "__main__":
    # 1. 检查 Python 版本
    check_python_version(3, 7)

    # 2. 检查需要的依赖包
    required_packages = [
        ("numpy", ("ge", "1.19.0")),      # 要求 numpy >= 1.19.0
        ("pandas", ("ge", "1.1.0")),     # 要求 pandas >= 1.1.0
        ("sklearn", None),               # sklearn 不检查版本
        ("transformers", None),          # transformers 不检查版本
        ("torch", ("ge", "1.8.0")),      # 要求 torch >= 1.8.0
    ]
    for pkg, v_check in required_packages:
        check_package_installed(pkg, v_check)

    # 3. 检查 GPU
    check_gpu_with_pytorch()

    print("所有环境检查完成！如果没有抛出错误，就可以继续后续步骤。")

    import torch

if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU.")
