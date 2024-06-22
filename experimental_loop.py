from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
import pandas as pd

from utils.CPU_utils import get_cpu_metrics
from utils.GPU_utils import get_gpu_metrics, initialize_correct_pynvml_device, clear_gpu_cache
import gc
import time
from utils.experiment_utils import WARM_UP_ITERATIONS, EXPERIMENT_ITERATIONS, models_dict, prompts_dict, PROMPT_PREFIX, \
    SAVE_LOCATION_PREFIX, inference_steps, WARM_UP_STEPS, GUIDANCE_SCALE, WARM_UP_PROMPT
from utils.logger import logger

handle = initialize_correct_pynvml_device()

for model_info in models_dict.items():
    logger.info("Running benchmark for model %s ", model_info[0])
    
    pipe = AutoPipelineForText2Image.from_pretrained(model_info[1], torch_dtype=torch.float16, variant="fp16")
    pipe.to('cuda')

    logger.info("Warming up model")
    for _ in range(WARM_UP_ITERATIONS):
        gc.collect()
        torch.cuda.empty_cache()
        _ = pipe(prompt=(PROMPT_PREFIX + WARM_UP_PROMPT), num_inference_steps=WARM_UP_STEPS,
                 guidance_scale=GUIDANCE_SCALE).images[0]

    logger.info("Warm up complete")

    for prompt_info in prompts_dict.items():
        logger.info("Running benchmark for model %s prompt %s", model_info[0], prompt_info[0])
        for inference_step in inference_steps:
            logger.info("Inferece step %d", inference_step)

            metrics = []
            start_time = time.time()

            for i in range(EXPERIMENT_ITERATIONS):
                logger.info("Running iteration %d", i)
                
                clear_gpu_cache()

                start_infer_time = time.time()
                image = \
                    pipe(prompt=PROMPT_PREFIX + prompt_info[1], num_inference_steps=inference_step,
                         guidance_scale=GUIDANCE_SCALE).images[0]
                end_infer_time = time.time()
                
                iteration_metrics = get_gpu_metrics(handle)
                cpu_metrics = get_cpu_metrics()
                iteration_metrics.update(cpu_metrics)
                iteration_metrics['inference_time'] = end_infer_time - start_infer_time
                metrics.append(iteration_metrics)

                image_dir = Path(SAVE_LOCATION_PREFIX,
                                 f'resulting_images/{model_info[0]}/{prompt_info[0]}/step_{inference_step}')
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = Path(image_dir,
                                  f'{model_info[0]}_{prompt_info[0]}_step-{inference_step}_iter-{i + 1}.png')
                image.save(image_path)

            end_time = time.time()
            logger.info("Benchmarking loop complete")
            print(f"Total benchmarking time: {end_time - start_time:.2f} seconds")

            df_metrics = pd.DataFrame(metrics)
            csv_path = Path(SAVE_LOCATION_PREFIX, 'partial_csvs'
                            , f'{model_info[0]}_{prompt_info[0]}_step-{inference_step}.csv')
            df_metrics.to_csv(csv_path, index=False)
            print(df_metrics.describe())
