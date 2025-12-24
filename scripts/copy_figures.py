from pathlib import Path
import shutil
src = Path('literature_comparison/comparison')
dst = Path('static/figures')
dst.mkdir(parents=True, exist_ok=True)
for p in src.glob('*.png'):
    shutil.copy(p, dst / p.name)
print('copied:', [p.name for p in dst.glob('*.png')])
