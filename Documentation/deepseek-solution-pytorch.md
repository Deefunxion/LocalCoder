Yes, you can get PyTorch working with your RTX 5070 Ti on Windows right now. Official support for the Blackwell architecture (sm_120) is available in the latest PyTorch nightly and stable versions that are built with CUDA 12.8 or newer.

Based on your questions, here is a detailed, actionable guide.

### ✅ Official PyTorch Support and Installation

PyTorch now provides native support for Blackwell GPUs like your RTX 5070 Ti through specific builds.

- **Supported Builds**: You need a PyTorch version compiled with **CUDA 12.8** or later. This support is available in:
    - **Stable Release**: PyTorch **v2.7.0** and above with CUDA 12.8.
    - **Nightly Builds**: All nightly builds with `cu128` in their version name.
- **Verification**: A PyTorch developer confirms that in a correct installation, `torch.cuda.get_arch_list()` should output a list that includes `sm_120`.

#### Step-by-Step Installation Guide

Follow these steps to set up a compatible environment. The most critical step is ensuring a clean installation of the correct PyTorch library.

1.  **Create a Clean Python Environment**: This prevents conflicts with older PyTorch installations or other packages.
    ```bash
    # Using conda
    conda create -n pytorch_blackwell python=3.10 -y
    conda activate pytorch_blackwell

    # Or, using venv
    python -m venv pytorch_blackwell
    .\pytorch_blackwell\Scripts\activate
    ```

2.  **Install the Correct PyTorch Build**: Use the official pip command to install the stable version with CUDA 12.8 support.
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
    *Note: As of early November 2025, the official stable wheel works for at least one user with your exact hardware and OS configuration.*

3.  **Verify Your Installation**: Run a comprehensive check to confirm everything is working.
    ```python
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Available GPU architectures: {torch.cuda.get_arch_list()}")
    
    # Simple GPU computation test
    if torch.cuda.is_available():
        x = torch.randn(3, 3).cuda()
        y = torch.matmul(x, x)
        print(f"GPU test successful. Result shape: {y.shape}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    ```
    A successful output will show `sm_120` in the architectures and complete the computation without errors.

#### Troubleshooting Common Hurdles

If you still face issues, they are likely due to your environment.

- **Problem: Mixed Binaries**: Other software you are using (like Stable Diffusion WebUI or ComfyUI) might automatically install an older, incompatible version of PyTorch, creating a conflict in your environment.
    - **Solution**: Always activate your clean `pytorch_blackwell` environment before running your projects. Verify the PyTorch version from within your project's script or environment.

- **Problem: Long Path Installation Error**: During installation, you might encounter a `"[WinError 206] The filename or extension is too long"`.
    - **Solution**: Use the `--no-cache-dir` flag with pip or set your temporary directory to a shorter path.
    ```bash
    # Method 1: Disable cache
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

    # Method 2: Use a short temp path (run before pip install)
    set TMP=D:\tmp
    set TEMP=D:\tmp
    ```

### 🚧 Alternative Frameworks and Workarounds

If you need a temporary solution or want to explore other options, here are alternatives.

- **ONNX Runtime with CUDA**: This is an excellent alternative for running inference on models like sentence-transformers.
    - **Setup**: Ensure you have the latest version of ONNX Runtime with the CUDA execution provider. It requires a Blackwell-compatible driver (R570 or higher) and will work out-of-the-box.
    - **Usage**: You can convert your PyTorch models to ONNX format and then run them using ONNX Runtime for GPU-accelerated inference.

- **TensorFlow**: For embedding generation, TensorFlow is a viable alternative. Make sure to install a version built with CUDA 12.8 or higher for native sm_120 support.

- **Optimized CPU Performance**: If you must rely on the CPU, you can mitigate performance loss by:
    - **Batch Size Tuning**: Experiment with smaller batch sizes that fit your CPU's memory bandwidth.
    - **Multi-threading**: Leverage PyTorch's built-in CPU parallelization by setting `torch.set_num_threads()` to the number of your CPU cores.
    - **Library-Specific Optimizations**: Using libraries like Intel Extension for PyTorch (IPEX) can further accelerate performance on Intel CPUs.

### 🗓️ Timeline and Official Stance

You are not dealing with a future promise but with currently available technology.

- **Official Support Status**: Support for Blackwell (sm_100 and sm_120) in PyTorch is no longer experimental. It has been available in nightly builds since early 2025 and is now part of the stable release channel.
- **Roadmap**: The current stable version (2.7.0+) with CUDA 12.8 is the officially supported path for RTX 50-series GPUs. There is no need to wait for a future version like PyTorch 2.10 or 3.0 for basic support.

### 💡 Your Action Plan

1.  **First and Recommended**: Follow the **Step-by-Step Installation Guide** above in a clean environment. This is the most straightforward path to getting your RTX 5070 Ti working with PyTorch.
2.  **If Issues Persist**: Meticulously check for conflicting PyTorch installations using `pip list | findstr torch` and uninstall any versions that are not `cu128`.
3.  **As a Temporary Measure**: While resolving PyTorch setup, you can use **ONNX Runtime** for GPU acceleration of your embedding models.
4.  **For Maximum CPU Performance**: Implement the CPU optimization techniques mentioned if GPU setup is delayed.

I hope this detailed guide helps you get your system up and running at full speed. If you encounter a specific error message during the installation or verification process, feel free to share it for more targeted troubleshooting.