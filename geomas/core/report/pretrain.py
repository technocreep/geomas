import torch
from geomas.core.logger import get_logger


logger = get_logger()


def pretrain_report(
        
):
    logger.info("Pre-Training report >>> ")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    return start_gpu_memory, max_memory


def posttrain_report(
    start_gpu_memory, max_memory, trainer_stats
):
    logger.info("Post-Training report >>>")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    train_runtime = trainer_stats.metrics['train_runtime']
    train_runtime_min = round(trainer_stats.metrics['train_runtime']/60, 2)
    logger.info(f"{train_runtime} seconds used for training.")
    logger.info(f"{train_runtime_min} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    return {"train_runtime_min": train_runtime_min,
            "peak_reserved_memory": used_memory,
            "peak_reserver_memory_training": used_memory_for_lora}
