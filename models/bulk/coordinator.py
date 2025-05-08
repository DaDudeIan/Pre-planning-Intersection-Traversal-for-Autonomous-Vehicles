import subprocess
import time
import os
from datetime import datetime

# List of (description, script filename) pairs
scripts = [
    #("Simple test", "train_deeplab_dummy.py"),
    #("DeepLabV3+ (CE)", "train_deeplab_ce.py"), # ✔️
    #("U-Net (CE)", "train_unet_ce.py"), # ✔️
    #("ViT (CE)", "train_vit_ce.p✔️
    #("Swin (CE)", "train_swin_ce.py"), # ✔️
    
    #("DeepLabV3+ (CE+Topo)", "train_deeplab_ce-topo.py"), # ✔️
    #("U-Net (CE+Topo)", "train_unet_ce-topo.py"), # ✔️
    #("ViT (CE+Topo)", "train_vit_ce-topo.py"), # ✔️
    #("Swin (CE+Topo)", "train_swin_ce-topo.py"), # ✔️
    
    
    ("DeepLabV3+ (CE+cmap)", "train_deeplab_ce-cmap.py"), # ✔️, but redo with better alpha
    ("U-Net (CE+cmap)", "train_unet_ce-cmap.py"), # ✔️, but redo with better alpha
    ("ViT (CE+cmap)", "train_vit_ce-cmap.py"), # ✔️, but redo with better alpha
    ("Swin (CE+cmap)", "train_swin_ce-cmap.py"), # ✔️, but redo with better alpha
    
    ("DeepLabV3+ (cmap)", "train_deeplab_cmap.py"), # ✔️, but redo with lower threshold
    ("U-Net (cmap)", "train_unet_cmap.py"), # ✔️, but redo with lower threshold
    ("ViT (cmap)", "train_vit_cmap.py"), # ✔️, but redo with lower threshold
    ("Swin (cmap)", "train_swin_cmap.py"), # ✔️, but redo with lower threshold
]

# Assert all scripts exist in the current directory
for _, script in scripts:
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Script '{script}' not found in the current directory.")

max_retries = 2  # Number of retries allowed if a model fails

failed_models = []

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Create a timestamped log file
log_file_path = os.path.join("logs", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

print("\n=== Starting coordinated training ===\n")
start_time = time.time()

# Open the log file
with open(log_file_path, "w") as log_file:

    for desc, script in scripts:
        attempt = 0
        success = False
        
        while attempt <= max_retries and not success:
            attempt += 1
            print(f"Training: {desc} (Attempt {attempt}/{max_retries + 1})")
            print("-" * 50)
            log_file.write(f"Training: {desc} (Attempt {attempt}/{max_retries + 1})\n")
            log_file.write("-" * 50 + "\n")

            # Start subprocess
            process = subprocess.Popen(["python3", "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Stream output to console and log
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    log_file.write(output)
                    log_file.flush()

            # Check exit status
            exit_code = process.poll()
            if exit_code == 0:
                print(f"✅ Training finished successfully for {desc} (Attempt {attempt})\n")
                log_file.write(f"✅ Training finished successfully for {desc} (Attempt {attempt})\n\n")
                success = True
            else:
                print(f"⚠️ Training FAILED for {desc} (Attempt {attempt})\n")
                log_file.write(f"⚠️ Training FAILED for {desc} (Attempt {attempt})\n\n")
                if attempt > max_retries:
                    print(f"❌ Maximum retries exceeded for {desc}. Marking as FAILED.\n")
                    log_file.write(f"❌ Maximum retries exceeded for {desc}. Marking as FAILED.\n\n")
                    failed_models.append(desc)

    print("=" * 50)
    print("\n=== Training Summary ===")
    log_file.write("=" * 50 + "\n")
    log_file.write("\n=== Training Summary ===\n")
    
    if failed_models:
        print("Failed models:")
        log_file.write("Failed models:\n")
        for model in failed_models:
            print(f"- {model}")
            log_file.write(f"- {model}\n")
    else:
        print("All models trained successfully! 🎉")
        log_file.write("All models trained successfully! 🎉\n")

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTotal time elapsed: {total_minutes:.2f} minutes\n")
    log_file.write(f"\nTotal time elapsed: {total_minutes:.2f} minutes\n")

print(f"\nFull logs saved at: {log_file_path}")
