import fire

# from huggingface_hub import snapshot_download, load_from_checkpoint
from comet import download_model, load_from_checkpoint

model2path = {
    "Unbabel/wmt22-cometkiwi-da": "/home/youyuan/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt",
    "Unbabel/wmt23-cometkiwi-da-xl": "/home/youyuan/.cache/huggingface/hub/models--Unbabel--wmt23-cometkiwi-da-xl/snapshots/247f80c250e569fb011dbd906af24f8afe3e8d58/checkpoints/model.ckpt"
}

def std(arr):
    return sum([(i - sum(arr) / len(arr)) ** 2 for i in arr]) ** 0.5

def comet_ref_free(
        data:list,
        model_path = "Unbabel/wmt22-cometkiwi-da", # "Unbabel/wmt23-cometkiwi-da-xl"
        batch_size = 16,
        gpus = 1
    ):
    """
    data: [
        {
            "src": "Hello",
            "mt": "Hallo"
        },
        ...
    ]
    """
    # mp = download_model(model_path)
    # print(model_path)
    # model_checkpoint_path = mp
    model_checkpoint_path = model2path[model_path]
    default_model = load_from_checkpoint(model_checkpoint_path)
    res = default_model.predict(data, batch_size=batch_size, gpus=gpus)
    return res['scores'], res['system_score']

def comet(
        srcs,
        mts,
        model:str = "Unbabel/wmt22-cometkiwi-da",
    ):
    data = []
    for ind, (src, mt) in enumerate(zip(srcs, mts)):
        data.append({"src": src, "mt": mt})
    comet_res, comet_avg = comet_ref_free(data, model)
    
    return comet_res, comet_avg

def eval(
        srcs,
        mts,
    ):
    return 1,1

def main(
        model:str = "Unbabel/wmt22-cometkiwi-da",
        src_file:str = "./datasets/src_test_190.txt", 
        mt_file:str = "./datasets/mt_test_190.txt", 
    ):

    data = []
    with open(src_file, "r") as f:
        src_lines = f.readlines()
    with open(mt_file, "r") as f:
        mt_lines = f.readlines()

    for ind, (src, mt) in enumerate(zip(src_lines, mt_lines)):
        data.append({"src": src, "mt": mt})
    comet_res, comet_avg = comet_ref_free(data, model)
    print(f"comet_avg: {comet_avg}, std: {std(comet_res)}")

if __name__ == "__main__":
    fire.Fire(main)