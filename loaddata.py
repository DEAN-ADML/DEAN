import numpy as np
import json


def loaddata():
    with open("hyper.json","r") as f:hyper=json.loads(f.read())
    if hyper["representation"]=="latent":
        f=np.load("datasets/latent.npz")
    elif hyper["representation"]=="mnist":
        f=np.load("datasets/mnist.npz")
    else:
        print("specify either latent or mnist as representation")
        exit()
    return (f["train_x"],f["train_y"]),(f["test_x"],f["test_y"])

if __name__=="__main__":
    (x,y),(tx,ty)=loaddata()

    print(x.shape)
