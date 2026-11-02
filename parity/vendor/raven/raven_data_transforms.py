import random

import torch


class AdaptiveLengthTimeMask:
    def __init__(self, window, stride, replace_with_zero=False, inplace=False):
        self.window = window
        self.stride = stride

        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, x):
        if self.inplace:
            cloned = x
        else:
            cloned = x.clone()

        len_raw = cloned.size(1)

        n_mask = int((len_raw + self.stride - 0.1) // self.stride)

        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if len_raw - t <= 0:
                continue
            t_start = random.randrange(0, len_raw - t)

            # avoids randrange error if values are equal and range is empty
            if t_start == t_start + t:
                continue

            t_end += t_start
            if self.replace_with_zero:
                cloned[:, t_start:t_end] = 0
            else:
                cloned[:, t_start:t_end] = cloned.mean()
        return cloned
