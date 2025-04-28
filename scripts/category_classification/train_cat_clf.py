import os
import subprocess
import glob

def run_training_for_all_categories():
    # Base directory containing category CSV files
    data_dir = "data/category_classification/category_csv"
    
    # Get all CSV files in the directory (excluding complete_data.csv)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f) != "complete_data.csv"]
    
    # Base command
    base_cmd = "python scripts/category_classification/model_training.py"
    
    # Common arguments
    common_args = "--model all --ratio 3 --epochs 50"
    
    # Create output directory for logs
    log_dir = "logs/training_runs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Run training for each category
    for csv_file in csv_files:
        # Get category name from file name
        category = os.path.basename(csv_file).replace(".csv", "")
        
        # Create experiment name
        experiment_name = f"{category}_V1_ratio_3"
        
        # Construct full command
        cmd = f"{base_cmd} --data_path {csv_file} --experiment_name {experiment_name} {common_args}"
        
        # Log file path
        log_file = os.path.join(log_dir, f"{category}_training.log")
        
        print(f"Starting training for {category}...")
        print(f"Command: {cmd}")
        print(f"Logging output to: {log_file}")
        
        # Run the command and redirect output to log file
        with open(log_file, "w") as f:
            process = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
        # Check if the process was successful
        if process.returncode == 0:
            print(f"Training completed successfully for {category}")
        else:
            print(f"Training failed for {category} with return code {process.returncode}")
        
        print("-" * 80)

if __name__ == "__main__":
    run_training_for_all_categories()