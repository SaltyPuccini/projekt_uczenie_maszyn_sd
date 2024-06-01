import os
from pathlib import Path

import pandas as pd

from experiment_utils import SAVE_LOCATION_PREFIX

results_of_experiments: dict = {
    'model': [],
    'prompt': [],
    'step': [],
    'guidance': [],
    'average_inference_time': [],
    'average_gpu_utilization': [],
    'average_memory_utilization': [],
    'average_cpu_percent': [],
    'average_cpu_user': [],
    'average_cpu_system': [],
    'average_cpu_idle': [],
}


def analyze_csv_file(file_path):
    global results_of_experiments
    # Load metrics from CSV
    df_metrics = pd.read_csv(file_path)
    filename = file_path.split('/')[-1]

    # Basic statistics
    print(f"{filename}:")

    # Example analysis: average inference time
    average_inference_time = df_metrics['inference_time'].mean()
    print(f"Average inference time: {average_inference_time:.4f} seconds")

    # GPU utilization analysis
    average_gpu_utilization = df_metrics['gpu_utilization'].mean()
    print(f"Average GPU utilization: {average_gpu_utilization:.2f}%")

    # Memory utilization analysis
    average_memory_utilization = df_metrics['memory_utilization'].mean()
    print(f"Average memory utilization: {average_memory_utilization:.2f}%")

    # CPU utilization analysis
    average_cpu_percent = df_metrics['cpu_percent'].mean()
    print(f"Average CPU utilization: {average_cpu_percent:.2f}%")

    # CPU user, system, and idle time analysis
    average_cpu_user = df_metrics['cpu_user'].mean()
    average_cpu_system = df_metrics['cpu_system'].mean()
    average_cpu_idle = df_metrics['cpu_idle'].mean()
    print(f"Average CPU user time: {average_cpu_user:.2f}%")
    print(f"Average CPU system time: {average_cpu_system:.2f}%")
    print(f"Average CPU idle time: {average_cpu_idle:.2f}%")

    print("\n")

    results_of_experiments['model'].append(filename.split('_')[0])
    results_of_experiments['prompt'].append(filename.split('_')[1])
    results_of_experiments['step'].append(filename.split('_')[2].split('-')[1])
    results_of_experiments['guidance'].append(filename.split('_')[3].split('-')[1].split('.')[0])
    results_of_experiments['average_inference_time'].append(round(average_inference_time, 4))
    results_of_experiments['average_gpu_utilization'].append(round(average_gpu_utilization, 2))
    results_of_experiments['average_memory_utilization'].append(round(average_memory_utilization, 2))
    results_of_experiments['average_cpu_percent'].append(round(average_cpu_percent, 2))
    results_of_experiments['average_cpu_user'].append(round(average_cpu_user, 2))
    results_of_experiments['average_cpu_system'].append(round(average_cpu_system, 2))
    results_of_experiments['average_cpu_idle'].append(round(average_cpu_idle, 2))


def analyze_directory(directory_path):
    # List all CSV files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Iterate over each file and analyze
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        analyze_csv_file(file_path)


analyze_directory(SAVE_LOCATION_PREFIX)
results = pd.DataFrame(results_of_experiments)
csv_path = Path(SAVE_LOCATION_PREFIX,
                f'STABLE_DIFFUSION_RESULTS.csv')
results.to_csv(csv_path, index=False)
