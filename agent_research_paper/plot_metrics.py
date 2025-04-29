import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(config):
    df = pd.read_csv('results/metrics.csv')
    plt.figure()
    plt.plot(df['Model_Size_MB'], df['WER'], marker='o')
    plt.xlabel('Model Size (MB)')
    plt.ylabel('WER (%)')
    plt.title('WER vs. Model Size')
    plt.grid()
    plt.savefig('results/plots/wer_vs_modelsize.png')
    print('Saved plot wer_vs_modelsize.png')

if __name__ == '__main__':
    import yaml, argparse
    args = argparse.ArgumentParser(); args.add_argument('--config', required=True); cfg = yaml.safe_load(open(args.parse_args().config))
    plot_metrics(cfg)
