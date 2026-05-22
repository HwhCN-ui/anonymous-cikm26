from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train(self, round, avg_local_loss, benign_cnt, byzantine_cnt):
        self.writer.add_scalar("Train/Avg_Local_Loss", avg_local_loss, round)
        self.writer.add_scalar("Train/Benign_Clients_Count", benign_cnt, round)
        self.writer.add_scalar("Train/Byzantine_Clients_Count", byzantine_cnt, round)

    def log_eval(self, round, benign_acc, asr=None):
        self.writer.add_scalar("Eval/Benign_Accuracy", benign_acc, round)
        if asr is not None:
            self.writer.add_scalar("Eval/Backdoor_ASR", asr, round)

    def log_lora(self, round, lora_norm):
        self.writer.add_scalar("LoRA/Params_Frobenius_Norm", lora_norm, round)

    def log_lorasf(self, round, metrics: dict):
        for k, v in metrics.items():
            self.writer.add_scalar(f"LoRA-SF/{k}", v, round)

    def close(self):
        self.writer.close()
