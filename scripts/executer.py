import sys
import time
import yaml
import subprocess

from pathlib import Path


current_dir = Path(__file__).parent
target_script = current_dir / "run_simulation.py"

with open("../config/recipe.yaml", "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)['settings']

with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

for current_settings, in settings:
    print(current_settings)
    config['minimize']['dim'] = current_settings['dim']

    config['minimize']['optimizer'] = current_settings['optimizer']

    config['optimization'][current_settings['optimizer']]['tol'] = current_settings['tol']

    config['minimize']['lambda_val'] = current_settings['lambda val']

    config['minimize']['overlap'] = current_settings['overlap']

    with open('../config/params.yaml', 'w') as f:
        yaml.dump(config, f)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(target_script)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    elapsed_time_raw = time.time() - start
    minutes, seconds = divmod((elapsed_time_raw), 60)
    time_str = f"{int(minutes)}m {seconds:.2f}s"
    if result.returncode == 0:
        print(f"✅ 실험 완료, 총 시간 {time_str}")
    else:
        print("❌ 실험 중 에러 발생:")
        print(result.stderr)
