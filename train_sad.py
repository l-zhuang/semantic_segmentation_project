class Train:
    def __init__(self, Model, data, optim, criterion, metric, device):
        self.model = Model
        self.data = data
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
    
    def run_epoch(self, iteration_loss=False,sad=False):
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            outputs,loss_distill = self.model(inputs)
            loss = self.criterion(outputs, labels)
            if sad:
                loss=loss+0.1*loss_distill
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
            self.metric.add(outputs.detach(), labels.detach())
            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
        return epoch_loss / len(self.data), self.metric.value()