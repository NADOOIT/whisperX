# ⚙️ Installation & Requirements

This guide covers how to install WhisperX Adaptive Learning and its requirements.

---

## 🛠️ System Requirements

- **Python:** 3.8 or newer
- **Operating System:** Linux, macOS, or Windows
- **Optional:** CUDA-capable GPU for faster processing
- **Recommended:** 4+ CPU cores, sufficient RAM

---

## 🚀 Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NADOOIT/whisperX.git
   cd whisperX
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements_adaptive.txt
   ```
3. **(Optional) Enable CUDA/GPU support:**
   - Install NVIDIA drivers and CUDA
   - Install `torch` with GPU support

> 💡 **Tip:** The default installation is sufficient for CPU-only usage.

---

## 🔄 Updating

To update the repository and dependencies:
```bash
git pull
pip install -r requirements_adaptive.txt --upgrade
```

---

## 🆘 Common Installation Issues & Solutions

| Error Message                | Cause                              | Solution                                    |
|-----------------------------|------------------------------------|---------------------------------------------|
| Profile not found            | Wrong/missing profile name          | Check name, create profile                  |
| Permission denied            | No write permission in profile dir  | Check permissions, try as admin             |
| CUDA device not found        | No/unrecognized GPU                 | Check CUDA/drivers, use CPU mode if needed  |
| Audio too short              | Speech sample too short             | Use longer, clearer sample                  |
| ImportError/ModuleNotFound   | Missing dependency                  | `pip install -r requirements_adaptive.txt`  |
| AssertionError in Test       | Package not correctly installed     | Check installation, reinstall if needed     |

> ℹ️ **More Help:**
> - [FAQ](./faq_en.md)
> - [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)
