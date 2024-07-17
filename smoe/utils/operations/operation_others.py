from types import NoneType

from .operation_string import str2bool, str2none


def auto_convert_args_to_none(args):
    args_dict = vars(args)
    for key, value in vars(args).items():
        if isinstance(value, (NoneType, str)):
            try:
                converted_value = str2none(value, extended=False)
                print(f"args.{key}: {args_dict[key]} ({type(args_dict[key])}) --> {converted_value} ({type(converted_value)})")
                setattr(args, key, converted_value)
            except:
                continue
    return args


def auto_convert_args_to_bool(args):
    args_dict = vars(args)
    for key, value in vars(args).items():
        if isinstance(value, (bool, str)):
            try:
                converted_value = str2bool(value, extended=False)
                print(f"args.{key}: {args_dict[key]} ({type(args_dict[key])}) --> {converted_value} ({type(converted_value)})")
                setattr(args, key, converted_value)
            except:
                continue
    return args
