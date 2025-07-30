from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn

class ARMTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        training_args = kwargs["args"]

        self.gamma = training_args.gamma # target_reward_margin
        self.length_normalization = training_args.length_normalization

        if self.length_normalization:
            print('\nUsing length normalization. This is not default for training Autoregressive RM and should only be used for testing purposes!\n')
        if self.gamma != 0 or self.length_normalization:
            print(f'\nARM Trainer: gamma = {self.gamma}, length_normalization = {self.length_normalization}\n')

    def arm_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the arm loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the arm loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        # 得分累加应该是通过这部分实现的
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits
        # import pdb
        # pdb.set_trace()
        all_logps, valid_length = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        if self.length_normalization:
            all_logps = all_logps / valid_length

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    # 只训练entropy最高的token
    # @staticmethod #这个关键字的意思是不需要self
    # def get_batch_logps(
    #     logits: torch.FloatTensor,
    #     labels: torch.LongTensor,
    #     label_pad_token_id: int = -100,
    #     is_encoder_decoder: bool = False,
    # ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    #     """Compute the log probabilities of the given labels under the given logits.

    #     Args:
    #         logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
    #         labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
    #         label_pad_token_id: The label pad token id.
    #         is_encoder_decoder: Whether the model is an encoder-decoder model.

    #     Returns:
    #         A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
    #     """

    #     if logits.shape[:-1] != labels.shape:
    #         raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    #     if not is_encoder_decoder:
    #         labels = labels[:, 1:].clone()
    #         logits = logits[:, :-1, :]
    #     loss_mask = labels != label_pad_token_id

    #     # dummy token; we'll ignore the losses on these tokens later
    #     labels[labels == label_pad_token_id] = 0

    #     per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    #     # entropy_keep_ratio = 0.8
    #     # # 计算熵的比率
    #     # if entropy_keep_ratio < 1.0:
    #     #     entropy = -torch.sum(F.softmax(logits.detach(), dim=-1) * F.log_softmax(logits.detach(), dim=-1), dim=-1)  # Shape: (batch_size, seq_len)
    #     #     entropy[~loss_mask] = float('-inf')  # Set invalid positions to infinity so they won't be selected

    #     #     # Calculate number of tokens to mask for each sample. clamp for safety
    #     #     num_tokens_to_mask = torch.clamp((loss_mask.sum(dim=-1) * entropy_keep_ratio).long(), min=1)  # Shape: (batch_size,)

    #     #     # Create entropy mask for each sample
    #     #     entropy_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    #     #     for i in range(entropy.shape[0]):
    #     #         _, largest_entropy_indices = torch.topk(entropy[i], k=num_tokens_to_mask[i])
    #     #         entropy_mask[i, largest_entropy_indices] = True

    #     #     loss_mask = loss_mask & entropy_mask

    #     return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the arm loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.arm_loss(
            policy_chosen_logps,
            policy_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics