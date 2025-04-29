import pandas as pd
import matplotlib.pyplot as plt

# Load attack metrics
attack = pd.read_csv('results/attack.csv')
fpr = attack['FAR']
tpr = 100 - attack['ASR']  # True Positive Rate as 100 - ASR

plt.figure()
plt.plot(fpr, tpr, marker='o')
for i, row in attack.iterrows():
    plt.annotate(row['Attack_Type'], (fpr[i], tpr[i]))
plt.xlabel('False Positive Rate (%)')
plt.ylabel('True Positive Rate (%)')
plt.title('Speaker Verification ROC Curve')
plt.grid()
plt.savefig('results/plots/roc_curve.png')
print('Saved ROC curve to results/plots/roc_curve.png')
plt.show()
