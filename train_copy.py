class Train:
    def __init__(self, Model, data, optim, metric, device):
        self.model = Model
        self.data = data
        self.optim = optim
        self.metric = metric
        self.device = device
    
    def run_epoch(self, iteration_loss=False,sad_loss=False):
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            outputs,loss_seg,loss = self.model(inputs,seg_gt=labels, sad_loss=sad_loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
            self.metric.add(outputs.detach(), labels.detach())
            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
        return epoch_loss / len(self.data), self.metric.value()