# Experiment Scripts & Plot Templates

This document contains example Python scripts and plotting templates for use in the experiments described in the paper "Towards Efficient and Adaptive Speaker-Specific Speech Codecs".

## 1. WER vs. Model Size Plot
```python
import matplotlib.pyplot as plt
cycles = [1,2,3,4]
wer = [12.8, 10.2, 8.7, 8.6]
size = [45, 36, 28, 22]
plt.plot(size, wer, marker='o')
plt.xlabel('Model Size (MB)')
plt.ylabel('WER (%)')
plt.title('WER vs. Model Size')
plt.grid()
plt.savefig('wer_vs_modelsize.png')
plt.show()
```

## 2. Speaker Verification ROC Curve
```python
import numpy as np
import matplotlib.pyplot as plt
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr)
plt.plot(fpr, tpr, label='LoRA-Adapter')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Speaker Verification ROC')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()
```

## 3. Attack Table Template
| Attack Type      | FAR (%) | ASR (%) | Mitigation                    |
|------------------|---------|---------|-------------------------------|
| Replay           | 0.7     | 1.1     | Challenge-response, liveness  |
| Deepfake         | 0.3     | 0.9     | High-dimensional LoRA traits  |
| Adversarial      | 1.5     | 2.2     | Adversarial training          |
| Device Loss      | 0       | 0       | Adapter revocation, recovery  |

## 4. Notes
- Replace dummy data with real experiment values.
- See paper Section 5 for context and interpretation.
