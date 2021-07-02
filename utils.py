import torch


def get_logger():
    return SimpleLogger()


class SimpleLogger(object):

    def trace(self, msg):
        print("===> [TRACE] " + msg)

    def debug(self, msg):
        print("[DEBUG] " + msg)

    def info(self, msg):
        print("[INFO] " + msg)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    output = torch.tensor(output)
    # target = torch.tensor(target)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
