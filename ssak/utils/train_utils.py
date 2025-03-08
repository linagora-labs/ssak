import os

from ssak.utils.misc import remove_commonprefix


def args_to_str(args, ignore=["gpus", "gpu"], sort=False):
    if not isinstance(args, dict):
        args = args.__dict__

    s = "_".join((f"{_short_name(k)}-{_short_value(v)}") for k, v in (sorted(args.items()) if sort else args.items()) if k not in ignore)
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _short_name(name, keep_if_shorter=4):
    if len(name) <= keep_if_shorter:
        return name.capitalize()
    if "-" in name or "_" in name:
        return "".join([_short_name(a, 3 if i == 0 else 3) for i, a in enumerate(name.replace("-", "_").split("_"))])
    return name[0].capitalize()


def _short_value(value):
    if isinstance(value, str) and "/" in value and os.path.exists(value):
        return dataset_pseudos(value)
    return {
        True: "1",
        False: "0",
        None: "",
    }.get(value, str(value).replace("/", "_"))


def dataset_pseudos(trainset, validset=None):
    return_two = validset is not None
    if validset is None:
        validset = trainset
    train_folders = sorted(trainset.split(","))
    valid_folders = sorted(validset.split(","))
    all_folders = train_folders + valid_folders
    all_folders = remove_commonprefix(all_folders, "/")
    train_folders = all_folders[: len(train_folders)]
    valid_folders = all_folders[len(train_folders) :]

    def base_folder(f):
        f = f.split("/")[0].split("\\")[0]
        if len(f.split("-")) > 1:
            f = "".join([s[0] for s in f.split("-")])
        return f

    train_base_folders = set(base_folder(f) for f in train_folders)
    valid_base_folders = set(base_folder(f) for f in valid_folders)
    train_folders = sorted(list(set([base_folder(f.replace("/", "_")) if base_folder(f) in valid_base_folders else base_folder(f) for f in train_folders])))
    valid_folders = sorted(list(set([base_folder(f.replace("/", "_")) if base_folder(f) in train_base_folders else base_folder(f) for f in valid_folders])))
    if return_two:
        return "t-" + "-".join(train_folders), "v-" + "-".join(valid_folders)
    else:
        return "-".join(train_folders)
