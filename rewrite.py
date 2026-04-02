import sys

with open('test_dataset_generalization.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_content = '''                else:
                    # === STANDARD BATCH FORMAT ===
                    num_samples = len(batch["latents"])
                    
                    # Prepare lists
                    batch_text_ids = []
                    for i in range(num_samples):
                        prompt_ids = batch["input_ids"][i].to(self.device, non_blocking=True)
                        if random.random() < 0.1:
                            batch_text_ids.append(empty_prompt_tokens(self.dataset.tokenizer, device=self.device))
                        else:
                            batch_text_ids.append(prompt_ids)
                    
                    # Prepare latents (list or stacked tensor)
                    latents_input = batch["latents"]
                    
                    if isinstance(latents_input, torch.Tensor):
                        latents_input = latents_input.to(self.device, dtype=self.dtype, non_blocking=True)
                        latents_list = [latents_input[i] for i in range(num_samples)]
                    else:
                        latents_list = [lat.to(self.device, dtype=self.dtype, non_blocking=True) if lat is not None else None for lat in latents_input]
                    
                    # Logit-Normal Timestep Sampling
                    if self.config.use_logit_normal_sampling:
                        t_tensor = sample_logit_normal(num_samples, mean=self.config.logit_normal_mean, std=self.config.logit_normal_std, device=self.device, dtype=self.dtype)
                    else:
                        t_tensor = torch.rand(num_samples, device=self.device, dtype=self.dtype)
                    
                    noisy_latents = []
                    v_targets = []
                    
                    for i in range(num_samples):
                        lat = latents_list[i]
                        if lat is not None:
                            if lat.dim() == 4 and lat.shape[0] == 1:
                                lat = lat.squeeze(0)
                        
                            noise = torch.randn_like(lat)
                            t_val = t_tensor[i]
                            x_t = (1.0 - t_val) * noise + t_val * lat
                            v_t = lat - noise
                            noisy_latents.append(x_t)
                            v_targets.append(v_t)
                        else:
                            noisy_latents.append(None)
                            v_targets.append(None)
                    
                    def compute_batch_loss():
                        res = self.model(batch_text_ids, noisy_latents, t_tensor, causal_text=True)
                        
                        pred_v = res["image"]
                        text_logits = res["text"]
                        mod_mask = res["modality_mask"]
                        cu_seqlens = res["cu_seqlens"]
                        
                        batch_total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                        sum_img_loss = 0.0
                        sum_txt_loss = 0.0
                        
                        for i in range(num_samples):
                            seq_start = cu_seqlens[i].item()
                            seq_end = cu_seqlens[i + 1].item()
                            
                            sample_mask = mod_mask[seq_start:seq_end]
                            sample_pred_v = pred_v[seq_start:seq_end]
                            sample_text_logits = text_logits[seq_start:seq_end]
                            
                            sample_v = sample_pred_v[sample_mask == 1.0]
                            sample_txt_logits = sample_text_logits[sample_mask == 0.0]
                            sample_txt_ids = batch_text_ids[i]
                            
                            ce_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                            if sample_txt_logits.shape[0] > 0:
                                shift_logits = sample_txt_logits[:-1]
                                shift_labels = sample_txt_ids.to(self.device)[1:]
                                min_len = min(shift_logits.shape[0], shift_labels.shape[0])
                                if min_len > 0:
                                    ce_loss = F.cross_entropy(shift_logits[:min_len], shift_labels[:min_len], ignore_index=PAD_TOKEN_ID)
                            sum_txt_loss += ce_loss.item()
                            
                            fm_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                            v_tgt = v_targets[i]
                            if v_tgt is not None and sample_v.shape[0] > 0:
                                p = patch_size
                                if v_tgt.dim() == 4: v_tgt = v_tgt.squeeze(0)
                                patches = v_tgt.unfold(1, p, p).unfold(2, p, p)
                                gh, gw = patches.shape[1], patches.shape[2]
                                target_flat = patches.permute(1, 2, 0, 3, 4).reshape(gh * gw, -1)
                                fm_loss = F.mse_loss(sample_v, target_flat)
                                
                                if self.config.use_min_snr_weighting:
                                    snr_weight = compute_min_snr_weight(t_tensor[i], gamma=5.0)
                                    fm_loss = fm_loss * snr_weight
                                fm_loss = fm_loss * self.config.lambda_img
                                sum_img_loss += fm_loss.item()
                            
                            alpha = self.config.alpha_ntp_text_only if (v_tgt is None) else self.config.alpha_ntp
                            sample_total_loss = alpha * ce_loss + fm_loss
                            batch_total_loss = batch_total_loss + sample_total_loss
                        
                        batch_total_loss = batch_total_loss / (num_samples * self.config.gradient_accumulation_steps)
                        return batch_total_loss, {"loss_img": sum_img_loss / num_samples, "loss_txt": sum_txt_loss / num_samples}

                    if self.use_amp:
                        with torch.amp.autocast('cuda', dtype=self.dtype):
                            loss, partial_metrics = compute_batch_loss()
                    else:
                        loss, partial_metrics = compute_batch_loss()
                    
                    # Backward
                    scaler.scale(loss).backward()
                    batch_loss += loss.item() * self.config.gradient_accumulation_steps
                    batch_metrics["loss_img"] += partial_metrics["loss_img"]
                    batch_metrics["loss_txt"] += partial_metrics["loss_txt"]
'''

new_lines = lines[:2314] + [new_content] + lines[2480:]

with open('test_dataset_generalization.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Rewrite complete.')
