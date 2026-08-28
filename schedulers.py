


class ConstantScheduler:

    def __init__(self, rate: float):

        self.rate = rate

    def __call__(self, epoch: int):

        return self.rate


schedulers_dict = {
    "constant": ConstantScheduler
}


def get_scheduler(function_name: str, **kwargs):

    return schedulers_dict[function_name](**kwargs)



