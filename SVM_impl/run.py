import sys
import subprocess

STEPS = [
    "feature_extraction.py",
    "preprocessing.py",
    "support_vectors.py"
]

def run_step(script):
    print(f"\n===== Running {script} =====")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error while running {script}. Pipeline stopped!")
        sys.exit(result.returncode)

def main():
    print("Starting ML pipeline...\n")
    for step in STEPS:
        run_step(step)
    
    print("\nPipeline completed successfully\n")

if __name__ == "__main__":
    main()
