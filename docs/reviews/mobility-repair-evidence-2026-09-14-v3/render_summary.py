"""Render measured campaign comparisons after the independent audit completes."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT/'audit.json').read_text())
    rows = [row for row in audit['comparisons'] if 'current' in row]
    if not rows:
        raise ValueError('No completed comparison rows to plot')
    labels = [f"{r['group'].title()}\n{r['seed']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout='constrained')
    for ax, key, ylabel in zip(axes, ['visual', 'collision'], ['Visual triangles (millions)', 'Collision triangles (millions)'], strict=True):
        before = [r['baseline']['triangles']/1e6 for r in rows]
        after = [(r['current']['triangles'] if key=='visual' else r['collider_output_triangles'])/1e6 for r in rows]
        ax.bar(x-.18, before, .36, color='#a7aeb8', label='Previous campaign')
        ax.bar(x+.18, after, .36, color='#247ba0', label='Required clearance + repair')
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT/'mesh_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
