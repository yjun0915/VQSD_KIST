import sys
import time
import yaml
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from pathlib import Path


class VQSDExecuterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KIST VQSD Experiment Executer")
        self.root.geometry("600x800")

        # 기본 경로 설정
        self.current_dir = Path(__file__).parent
        self.recipe_path = self.current_dir.parent / "config" / "recipe.yaml"
        self.params_path = self.current_dir.parent / "config" / "params.yaml"

        self.setup_ui()

    def setup_ui(self):
        # 파라미터 입력 구역
        input_frame = tk.LabelFrame(self.root, text="Experiment Settings (Use comma for lists)", padx=10, pady=10)
        input_frame.pack(fill="x", padx=10, pady=10)

        # 입력 필드 정의
        self.inputs = {}
        fields = [
            ("Dims", "3"),
            ("Lambda Vals", "300"),
            ("Overlaps", "0.75"),
            ("Optimizers", "COBYLA"),
            ("Tols", "0.01"),
            ("Q-Points", "0, 0.1875, 0.375, 0.5625, 0.75"),
            ("Trials", "10"),
            ("simulation/experiment", "experiment")
        ]

        for i, (label, default) in enumerate(fields):
            tk.Label(input_frame, text=label).grid(row=i, column=0, sticky="w")
            entry = tk.Entry(input_frame, width=40)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.inputs[label] = entry

        # 버튼 구역
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.run_btn = tk.Button(btn_frame, text="🚀 Run Experiments", command=self.start_experiment_thread,
                                 bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), padx=20)
        self.run_btn.pack(side="left", padx=10)

        # 로그 출력 구역
        tk.Label(self.root, text="Execution Logs:").pack(anchor="w", padx=10)
        self.log_area = scrolledtext.ScrolledText(self.root, height=25)
        self.log_area.pack(fill="both", padx=10, pady=5, expand=True)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def parse_inputs(self):
        try:
            dims = [int(x.strip()) for x in self.inputs["Dims"].get().split(",")]
            lambdas = [float(x.strip()) for x in self.inputs["Lambda Vals"].get().split(",")]
            overlaps = [float(x.strip()) for x in self.inputs["Overlaps"].get().split(",")]
            optimizers = [x.strip() for x in self.inputs["Optimizers"].get().split(",")]
            tols = [float(x.strip()) for x in self.inputs["Tols"].get().split(",")]
            q_points = [float(x.strip()) for x in self.inputs["Q-Points"].get().split(",")]
            trial = int(self.inputs["Trials"].get())
            return dims, lambdas, overlaps, optimizers, tols, q_points, trial
        except ValueError as e:
            messagebox.showerror("Input Error", f"입력값을 확인해주세요: {e}")
            return None

    def generate_recipe(self):
        data = self.parse_inputs()
        if not data: return False

        dims, lambdas, overlaps, optimizers, tols, q_points, trial = data
        configs = []

        for opt in optimizers:
            for d_idx, dim in enumerate(dims):
                for ov in overlaps:
                    for tol in tols:
                        configs.append({
                            'dim': dim, 'overlap': ov, 'lambda_val': lambdas[d_idx],
                            'optimizer': opt, 'tol': tol, 'q_points': q_points, 'trial': trial
                        })

        with open(self.recipe_path, 'w', encoding='utf-8') as f:
            yaml.dump({"settings": configs}, f)
        self.log(f"📝 Recipe generated with {len(configs)} combinations.")
        return True

    def start_experiment_thread(self):
        if self.generate_recipe():
            self.run_btn.config(state="disabled")
            thread = threading.Thread(target=self.run_experiments, daemon=True)
            thread.start()

    def run_experiments(self):
        try:
            with open(self.recipe_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)['settings']

            with open(self.params_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            total = len(settings)
            for i, current in enumerate(settings):
                self.log(
                    f"🔄 [{i + 1}/{total}] Running: Dim {current['dim']}, Ov {current['overlap']}, Tol {current['tol']}, Optmizer {current['optimizer']}...")

                # params.yaml 업데이트
                config['minimize'].update({
                    'dim': current['dim'],
                    'optimizer': current['optimizer'],
                    'lambda_val': current['lambda_val'],
                    'overlap': current['overlap'],
                    'q_points': current['q_points'],
                    'trial': current['trial']
                })
                config['optimization'][current['optimizer']]['tol'] = current['tol']

                with open(self.params_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f)

                # 서브프로세스 실행
                start_t = time.time()
                result = subprocess.run(
                    [sys.executable, str(self.current_dir / f"run_{self.inputs['simulation/experiment'].get()}.py")],
                    capture_output=True, text=True, encoding='utf-8'
                )

                elapsed = time.time() - start_t
                min_part, sec_part = divmod(elapsed, 60)

                if result.returncode == 0:
                    self.log(f"   ✅ Done! ({int(min_part)}m {sec_part:.2f}s)")
                else:
                    self.log(f"   ❌ Error at index {i}:\n{result.stderr}")

            self.log("\n✨ All experiments finished!")
            messagebox.showinfo("Success", "모든 실험이 완료되었습니다.")
        except Exception as e:
            self.log(f"🔥 Critical Error: {e}")
        finally:
            self.run_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = VQSDExecuterGUI(root)
    root.mainloop()