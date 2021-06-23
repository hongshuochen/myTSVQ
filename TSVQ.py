import numpy as np
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
from sklearn.cluster import KMeans

class TSVQ:
    def __init__(self, patch_size=4, stride=2, codebook_size=2048, random_state=0) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.codebook_size = codebook_size
        self.kmeans = [None]*(codebook_size-1)
        self.random_state = random_state
        self.max_layer = int(np.log2(self.codebook_size))

    def parent_idx(self, idx):
        return int((idx-1)/2)

    def left_child_idx(self, idx):
        return idx*2 + 1
    
    def right_child_idx(self, idx):
        return idx*2 + 2

    def layer(self, idx):
        return int(np.log2(idx+1))

    def fit(self, X):
        X = view_as_windows(X, (1,self.patch_size,self.patch_size, 1), (1,self.stride, self.stride, 1))
        X = X.reshape(-1, self.patch_size**2)
        print(X.shape)
        self.split(X,0)

    def split(self, X, idx):
        if self.layer(idx) > self.max_layer-1:
            return
        print("Index", idx, "Layer", self.layer(idx))
        kmeans = KMeans(n_clusters=2, n_init=2).fit(X)
        self.kmeans[idx] = kmeans
        left_X = X[kmeans.labels_ == 0]
        print("Left:", len(left_X))
        right_X = X[kmeans.labels_ == 1]
        print("Right:", len(right_X))
        self.split(left_X, self.left_child_idx(idx))
        self.split(right_X, self.right_child_idx(idx))

    def transform(self, X):
        X_shape = X.shape
        X = view_as_windows(X, (1,self.patch_size,self.patch_size, 1), (1,self.patch_size, self.patch_size, 1))
        block_shape = X.shape
        X = X.reshape(-1, self.patch_size**2)
        labels = []
        for i in range(len(self.kmeans)):
            labels.append(self.kmeans[i].predict(X)[...,np.newaxis])
        labels = np.concatenate(labels,axis=1)
        X = [self.get_codeword(l, 0) for l in labels]
        X = np.array(X)
        X = X.reshape(block_shape[0], block_shape[1], block_shape[2], self.patch_size, self.patch_size, 1)
        X = np.moveaxis(X, 3, 2)
        X = X.reshape(X_shape)
        return X
    
    def get_codeword(self, labels, idx):
        if self.layer(idx) == self.max_layer - 1:
            return self.kmeans[idx].cluster_centers_[labels[idx]]
        if labels[idx] == 0:
            return self.get_codeword(labels, self.left_child_idx(idx))
        else:
            return self.get_codeword(labels, self.right_child_idx(idx))

if __name__ == "__main__":
    import pickle
    import torch
    import matplotlib.pyplot as plt
    def show_results_2(images1, images2):
        figure = plt.figure(figsize=(16, 32))
        cols, rows = 12,4
        for i in range(1, cols * rows + 1, 2):
            sample_idx = torch.randint(len(images1), size=(1,)).item()
            img = images1[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.imshow(img, cmap="gray")
            plt.title("gt")
            plt.axis("off")
            img = images2[sample_idx]
            figure.add_subplot(rows, cols, i+1)
            plt.imshow(img, cmap="gray")
            plt.title("vq")
            plt.axis("off")
        figure.tight_layout()
        plt.show()

    def show_results(images1, images2, images3):
        figure = plt.figure(figsize=(16, 32))
        cols, rows = 9,4
        for i in range(1, cols * rows + 1, 3):
            # sample_idx = torch.randint(len(images1), size=(1,)).item()
            sample_idx = int(i/3)
            img = images1[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.imshow(img, cmap="gray")
            plt.title("gt")
            plt.axis("off")
            img = images2[sample_idx]
            rmse = round(np.average((images2[sample_idx]*10 - images1[sample_idx]*10)**2)**0.5,4)
            figure.add_subplot(rows, cols, i+1)
            plt.imshow(img, cmap="gray")
            plt.title("pred " + str(rmse))
            plt.axis("off")
            img = images3[sample_idx]
            rmse = round(np.average((images3[sample_idx]*10 - images1[sample_idx]*10)**2)**0.5,4)
            figure.add_subplot(rows, cols, i+2)
            plt.imshow(img, cmap="gray")
            plt.title("vq " + str(rmse))
            plt.axis("off")
        figure.tight_layout()
        plt.show()

    depth = pickle.load(open("data.pkl", "rb"))
    
    tsvq = TSVQ()
    tsvq.fit(depth)
    pickle.dump(tsvq, open("tsvq_4_2_2048.pkl","wb"))
    tsvq = pickle.load(open("tsvq_4_2_2048.pkl", "rb"))
    depth_vq = tsvq.transform(depth)
    show_results_2(depth, depth_vq)


    generated_depth = pickle.load(open("generated_dmaps.pkl", "rb"))
    idx = pickle.load(open("sort_idx.pkl", "rb"))
    generated_depth_vq = tsvq.transform(generated_depth)
    test_depth = pickle.load(open("test_data.pkl", "rb"))

    show_results(test_depth[idx[:12]], generated_depth[idx[:12]], generated_depth_vq[idx[:12]])
    def rmse_cal(images1, images2):
        images1= images1*10
        images2= images2*10
        images1 = images1.reshape(images1.shape[0], -1)
        images2 = images2.reshape(images2.shape[0], -1)
        avg = np.average((images1 - images2)**2, axis=1)
        avg = avg**0.5
        return np.average(avg)
    print(rmse_cal(test_depth, generated_depth))
    print(rmse_cal(test_depth, generated_depth_vq))
