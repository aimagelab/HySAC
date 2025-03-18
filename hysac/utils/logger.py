class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, **kwargs):
        self.run.log(kwargs)
    
    def log_recall_only_paired(self, recalls, recall_sum):
        S_V_recall, U_G_recall = recalls
        self.run.log(
            {
                'recall_S_V__t2i': S_V_recall[0][0],
                'recall_S_V__i2t': S_V_recall[1][0],
                'recall_U_G__t2i': U_G_recall[0][0],
                'recall_U_G__i2t': U_G_recall[1][0],
                'recall_sum': recall_sum
            }
        )

    def log_recall(self, recalls, recall_sum):
        S_V_recall, S_G_recall, U_V_recall, U_G_recall = recalls
        self.run.log(
            {
                'recall_S_V__t2i': S_V_recall[0][0],
                'recall_S_V__i2t': S_V_recall[1][0],
                'recall_S_G__t2i': S_G_recall[0][0],
                'recall_S_G__i2t': S_G_recall[1][0],
                'recall_U_V__t2i': U_V_recall[0][0],
                'recall_U_V__i2t': U_V_recall[1][0],
                'recall_U_G__t2i': U_G_recall[0][0],
                'recall_U_G__i2t': U_G_recall[1][0],
                'recall_sum': recall_sum
            }
        )
    
    def log_recall_frozen_clip(self, S_V_recall, recall_sum):
        self.run.log(
            {
                'recall_S_V__t2i': S_V_recall[0][0],
                'recall_S_V__i2t': S_V_recall[1][0],
                'recall_sum': recall_sum
            }
        )
    
    def log_training_iteration_A_entailment(self, idx, text_safe_loss, text_nsfw_loss, vision_safe_loss, vision_nsfw_loss, entail_A_safe, entail_A_nsfw, training_loss):
        self.run.log({
            'text_safe_training_loss': text_safe_loss.mean(),
            'text_nsfw_training_loss': text_nsfw_loss.mean(),
            'vision_safe_training_loss': vision_safe_loss.mean(),
            'vision_nsfw_training_loss': vision_nsfw_loss.mean(),
            'entail_a_safe_training_loss': entail_A_safe.mean(),
            'entail_a_nsfw_training_loss': entail_A_nsfw.mean(),
            'training_loss': training_loss,
            'batch_id': idx
        })

    def log_training_iteration(self, idx, text_safe_loss, text_nsfw_loss, vision_safe_loss, vision_nsfw_loss, S_Vref_contrastive_loss, U_Vref_contrastive_loss, V_Sref_contrastive_loss, G_Sref_contrastive_loss, training_loss):
        self.run.log({
            'text_safe_training_loss': text_safe_loss.mean(),
            'text_nsfw_training_loss': text_nsfw_loss.mean(),
            'vision_safe_training_loss': vision_safe_loss.mean(),
            'vision_nsfw_training_loss': vision_nsfw_loss.mean(),
            'S_Vref_contrastive_training_loss': S_Vref_contrastive_loss['loss'].mean(),
            'U_Vref_contrastive_training_loss': U_Vref_contrastive_loss['loss'].mean(),
            'V_Sref_contrastive_training_loss': V_Sref_contrastive_loss['loss'].mean(),
            'G_Sref_contrastive_training_loss': G_Sref_contrastive_loss['loss'].mean(),
            'training_loss': training_loss,
            'batch_id': idx
        })

    def log_training_iteration_custom(self, **kwargs):
        self.run.log(kwargs)

    def log_training_iteration_frozen_clip(self, idx, training_loss):
        self.run.log({
            'training_loss': training_loss,
            'batch_id': idx
        })
    
    def log_validation_A_entailment(self, len_validation_dataset, text_safe_loss_cumulative, text_nsfw_loss_cumulative, vision_safe_loss_cumulative, vision_nsfw_loss_cumulative, entail_A_safe_cumulative, entail_A_nsfw_cumulative, validation_loss, batch_size):
        self.run.log({
            'text_safe_validation_loss': text_safe_loss_cumulative / len_validation_dataset,
            'text_nsfw_validation_loss': text_nsfw_loss_cumulative / len_validation_dataset,
            'vision_safe_validation_loss': vision_safe_loss_cumulative / len_validation_dataset,
            'vision_nsfw_validation_loss': vision_nsfw_loss_cumulative / len_validation_dataset,
            'entail_A_safe_validation_loss': entail_A_safe_cumulative / len_validation_dataset,	
            'entail_A_nsfw_validation_loss': entail_A_nsfw_cumulative / len_validation_dataset,
            'validation_loss': validation_loss / batch_size,
        })
    
    def log_validation(self, len_validation_dataset, text_safe_loss_cumulative, text_nsfw_loss_cumulative, vision_safe_loss_cumulative, vision_nsfw_loss_cumulative, S_Vref_contrastive_loss_cumulative, U_Vref_contrastive_loss_cumulative, V_Sref_contrastive_loss_cumulative, G_Sref_contrastive_loss_cumulative, validation_loss, batch_size):
        self.run.log({
            'text_safe_validation_loss': text_safe_loss_cumulative / len_validation_dataset,
            'text_nsfw_validation_loss': text_nsfw_loss_cumulative / len_validation_dataset,
            'vision_safe_validation_loss': vision_safe_loss_cumulative / len_validation_dataset,
            'vision_nsfw_validation_loss': vision_nsfw_loss_cumulative / len_validation_dataset,
            'S_Vref_contrastive_validation_loss': S_Vref_contrastive_loss_cumulative / len_validation_dataset,
            'U_Vref_contrastive_validation_loss': U_Vref_contrastive_loss_cumulative / len_validation_dataset,
            'V_Sref_contrastive_validation_loss': V_Sref_contrastive_loss_cumulative / len_validation_dataset,
            'G_Sref_contrastive_validation_loss': G_Sref_contrastive_loss_cumulative / len_validation_dataset,
            'validation_loss': validation_loss / batch_size,
        })
    
    def log_validation_custom(self, **kwargs):
        self.run.log(kwargs)
    
    def log_validation_frozen_clip(self, validation_loss, batch_size):
        self.run.log({
            'validation_loss': validation_loss / batch_size,
        })

    def log_patience(self, patience):
        self.run.log({'patience': patience})

    def finish(self):
        self.run.finish()

def summarize(epoch, patience, training_loss, this_validation_loss, this_recalls, best_recall_sum, best_validation_loss, training_time, validation_time, best_checkpoint_saving_path):
    epoch_summary = f'''
*************** Epoch {epoch} ***************
patience: {patience}
############### Training Loss ###############
training_loss: {training_loss}
#############################################

############### Validation Loss ###############
this_validation_loss: {this_validation_loss}
###############################################

############### Recall ###############
S_V-recall: {this_recalls[0]}
V_S-recall: {this_recalls[1]}
##########################################

################## Best ###################
best_recall_sum: {best_recall_sum}
best_validation_loss: {best_validation_loss}

############### Timings ###############
training-time: {training_time}
validation-time: {validation_time}
##########################################

############### Checkpoint ###############
best_checkpoint_saving_path: {best_checkpoint_saving_path}
*********************************************
'''
    print(epoch_summary)