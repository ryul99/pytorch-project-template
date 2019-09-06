import numpy as np
import subprocess
import torch.nn.functional as F


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


def cosine_similarity(vector_list1, vector_list2):
    vector1_size, dim1 = vector_list1.size()
    vector2_size, dim2 = vector_list2.size()
    assert dim1 == dim2
    sizes = vector1_size, vector2_size, dim1
    similarity = F.cosine_similarity(
        vector_list1.unsqueeze(1).expand(sizes),
        vector_list2.unsqueeze(0).expand(sizes),
        2
    )
    return similarity
