# 🛠️ Troubleshooting

Find solutions for common problems with WhisperX Adaptive Learning.

---

## 🚫 Problem: Installation fails

**Error message:** `No module named whisperx`

> **Solution:**
> - Check if `pip install whisperx` was successful
> - Try a fresh Python environment (e.g. with `venv`)

---

## 🚫 Problem: Model does not load

**Error message:** `FileNotFoundError` or `Model not found`

> **Solution:**
> - Check the model name (`base`, `small`, ...)
> - Make sure you have internet access (for model download)

---

## 🚫 Problem: Speaker profile not recognized

> **Solution:**
> - Is the voice sample too short or noisy? Record a new sample!
> - Delete and recreate the profile if needed

---

## 🔎 More Tips
- Use `--verbose` for more log output
- See also [FAQ](./faq_en.md) and [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)

---

## Common Error Messages

| Error Message                | Cause                             | Solution                                    |
|-----------------------------|-----------------------------------|---------------------------------------------|
| Profile not found            | Wrong/missing profile name         | > Check name, create profile                  |
| Permission denied            | No write permission in profile dir | > Check permissions, try as admin             |
| CUDA device not found        | No/unrecognized GPU                | > Check CUDA/drivers, use CPU mode if needed  |
| Audio too short              | Speech sample too short            | > Use longer, clearer sample                  |
| ImportError/ModuleNotFound   | Missing dependency                 | > ```bash
> pip install -r requirements_adaptive.txt
> ``` |
| AssertionError in Test       | Package not correctly installed    | > Check installation, reinstall if needed     |
| Permission denied            | No write permission in profile dir | Check permissions, try as admin             |
| CUDA device not found        | No/unrecognized GPU                | Check CUDA/drivers, use CPU mode if needed  |
| Audio too short              | Speech sample too short            | Use longer, clearer sample                  |
| ImportError/ModuleNotFound   | Missing dependency                 | `pip install -r requirements_adaptive.txt`  |
| AssertionError in Test       | Package not correctly installed    | Check installation, reinstall if needed     |

## Debugging Tips
- Enable logs with `--verbose`
- Read Python tracebacks carefully
- Use the community/forum if stuck

## More Help
- [FAQ](./faq_en.md)
- [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)
