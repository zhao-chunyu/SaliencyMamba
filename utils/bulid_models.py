from .models.SalMM import SalMM


def build_model(args=None):
    if args.seq_len == 1:
        if args.network == 'salmm':
            model = SalMM()

    return model
