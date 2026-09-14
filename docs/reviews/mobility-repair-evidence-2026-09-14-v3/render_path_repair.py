"""Plot measured bounds and repaired placement through the same source mesh."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[2]


def main():
    with np.load(REPO/'tests/fixtures/geometry/route_placement_seed42.npz') as data:
        midpoint, lower, upper = data['path'], data['lower'], data['upper']
    report = json.loads((EVIDENCE/'production_path_repair.json').read_text())
    repaired = np.array(report['paths'][0]['center_path_m'])[128:151]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(midpoint[:, :2], axis=0), axis=1))]
    datum = float((lower-.27).min())
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    ax.fill_between(distance, lower-.27-datum, upper+.27-datum, color='#e6ebed', label='Measured cavity')
    ax.plot(distance, lower-.27-datum, color='#556572', linewidth=1.4)
    ax.plot(distance, upper+.27-datum, color='#556572', linewidth=1.4)
    ax.plot(distance, midpoint[:, 2]-datum, '--', color='#b63e31', linewidth=1.7, label='Rejected midpoint path')
    ax.plot(distance, repaired[:, 2]-datum, color='#147e6a', linewidth=2, label='Independently verified path')
    ax.set(xlabel='Horizontal distance along the sampled route (m)',
           ylabel='Height above local reference (m)')
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=9)
    ax.spines[['right','top']].set_visible(False)
    ax.grid(alpha=.15)
    fig.savefig(EVIDENCE/'path_repair.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
