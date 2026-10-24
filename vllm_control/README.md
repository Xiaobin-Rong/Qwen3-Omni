# vllm_control

当前已跑通环境中 **vLLM-Omni 0.30.0** 的源码快照，来源为 `site-packages/vllm_omni`。内部保留 `vllm_omni` 包名，避免破坏注册、动态导入和多进程启动路径。

- 复制了全部运行包代码和资源，排除 `__pycache__`、`.pyc`、`.pyo`。
- `SOURCE_MANIFEST.json` 记录来源、版本和逐文件 SHA-256；复制时与源文件逐一核对。
- `upstream_licenses/` 保留安装包携带的许可文件。
- 这是已安装运行包的快照，不是完整 Git checkout；不包含上游未打包的测试、构建脚本或历史。
- 尚未修改任何引擎实现，现有 `../inference.py` 也没有切换到该副本。
- 仍需已安装且版本匹配的 vLLM、PyTorch、CUDA 和其他依赖，不复制模型权重。

## 选择本地副本

在 `Qwen3-Omni` 目录中：

```bash
conda activate vllm
export PYTHONPATH="$PWD/vllm_control${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import vllm_omni; print(vllm_omni.__file__)'
```

输出必须指向本目录下的 `vllm_omni/__init__.py`。之后从同一终端运行 `inference.py` 时，父进程和继承该环境的工作进程都会优先导入副本。原 site-packages 不会被修改。已经启动的引擎需要重新启动才能采用修改。

控制方案与修改位置见 [CONTROL_DESIGN.md](CONTROL_DESIGN.md)。
