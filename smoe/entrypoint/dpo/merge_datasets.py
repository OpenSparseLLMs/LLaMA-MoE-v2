from datasets import DatasetDict, concatenate_datasets, load_dataset

ALL_DATASETS = {
    "argilla/ultrafeedback-binarized-preferences-cleaned": {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    },  # 60.9k
    # "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned": {
    #     "prompt": "prompt",
    #     "chosen": "chosen",
    #     "rejected": "rejected",
    # },  # 155k, an augmented version of the above dataset
    "argilla/distilabel-capybara-dpo-7k-binarized": {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    },  # 7.56k
    # "argilla/distilabel-intel-orca-dpo-pairs": {
    #     "input": "prompt",
    #     "chosen": "chosen",
    #     "rejected": "rejected",
    # },  # 12.9k, do not use as the format is incompatible!
}

save_path = "/mnt/petrelfs/dongdaize.d/workspace/llama-moe-v2/data/combined"

if __name__ == "__main__":
    all_ds = []
    all_keys = set()

    for dataset_name in ALL_DATASETS.keys():
        ds = load_dataset(dataset_name)
        print(dataset_name, ds.num_rows, ds.column_names)

        # for key in list(ds["train"].column_names):
        #     if key in ALL_DATASETS[dataset_name].keys():
        #         if key != ALL_DATASETS[dataset_name][key]:  # unify the keys
        #             ds["train"] = ds["train"].rename_column(key, ALL_DATASETS[dataset_name][key])
        #     else:
        #         ds["train"] = ds["train"].remove_columns(key)
        # print(dataset_name, ds.num_rows, ds.column_names, "\n")

        all_ds.append(ds)
        all_keys.update(ds.keys())

    cat_ds = DatasetDict(
        {"train": concatenate_datasets([ds["train"] for ds in all_ds])}
    )
    print("cat_dataset", cat_ds.num_rows, cat_ds.column_names)

    cat_ds.save_to_disk(save_path)
    print("Saved dataset to disk at", save_path)
