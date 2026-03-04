import sys
import yaml
import subprocess

from pathlib import Path


current_dir = Path(__file__).parent
target_script = current_dir / "run_simulation.py"

with open("../config/recipe.yaml", "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)['settings']

with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

exp_nums = settings.keys()

for exp_num in exp_nums:
    current_settings = settings[exp_num]
    for _ in config['flags']['activate'].keys():
        config['flags']['activate'][_] = False
    config['flags']['activate'][current_settings[0]] = True

    config['minimize']['optimizer'] = current_settings[1]

    config['optimization'][current_settings[1]]['tol'] = current_settings[2]

    config['minimize']['lambda_val'] = current_settings[3]

    with open('../config/params.yaml', 'w') as f:
        yaml.dump(config, f)
    result = subprocess.run(
        [sys.executable, str(target_script)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    if result.returncode == 0:
        print("✅ 실험 완료!")
    else:
        print("❌ 실험 중 에러 발생:")
        print(result.stderr)
