# Known Issues

This document tracks known issues, limitations, and temporary workarounds in Eloquent.

---

## Parakeet / NeMo Toolkit Installation Issues (NumPy Version Conflict)

When Eloquent is first installed, **Whisper** is selected by default as the Speech Recognition Engine.

If **Parakeet** is selected, Eloquent will usually report that Parakeet is not installed and attempt to install it automatically. Users can also trigger this manually using the **Force Install Parakeet** button.

In many cases, Parakeet will appear to fail installation and may report an error. At this point, Parakeet often does not appear to work.

However, after correcting the NumPy version, Parakeet will often work correctly on the next startup, even if it previously reported a failed installation.

### Cause

Parakeet depends on NVIDIA’s NeMo Toolkit. During installation, NeMo may replace the existing NumPy version with NumPy 2.x.

Some NeMo dependencies do not currently function correctly with NumPy 2.x. This can cause Parakeet to fail, crash, or behave unpredictably.

### Observed Behavior

- Parakeet may report that it failed to install
- Parakeet may not work immediately after installation
- After downgrading NumPy to a version below 2 and restarting Eloquent, Parakeet often works correctly
- Dependency warnings may appear after downgrading NumPy, but these do not appear to affect functionality

### Workaround

If Parakeet fails to install or does not work:

1. Close Eloquent
2. Open a new terminal
3. Activate the same virtual environment used by Eloquent
4. Run: pip install "numpy<2"
5. Restart Eloquent
6. Select Parakeet again if necessary

Even if Parakeet previously reported that installation failed, it may function correctly after this step.

---

Additional issues and workarounds will be documented here as they are discovered.

## AMD GPU Support Limitations on Windows

Eloquent officially supports **NVIDIA GPUs** (via CUDA) and **CPU** inference on Windows.

**AMD GPUs** are not currently supported for local inference due to the lack of pre-built Windows binaries (wheels) for the underlying inference libraries (`llama-cpp-python` and `stable-diffusion-cpp-python` with ROCm support).

### Workaround for AMD Users

AMD users have two options:
1.  **CPU Mode**: Run the application in default CPU mode. Eloquent uses capable quantized models that offer reasonable performance on modern CPUs.
2.  **External APIs**: Use the **OpenAI-Compatible** settings to connect Eloquent to an external inference server (like **Ollama** or **LM Studio**) running on your machine. These third-party tools maintain their own AMD-optimized builds and can provide GPU acceleration that Eloquent connects to as a client.