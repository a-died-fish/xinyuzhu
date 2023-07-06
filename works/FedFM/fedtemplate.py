import torch
import torch.nn.functional as F

def l2_loss(features, labels, centroids):
    # features = torch.nn.functional.normalize(features)
    # centroids = torch.nn.functional.normalize(centroids)

    centroids_new=centroids.unsqueeze(0).repeat(features.shape[0],1,1)
    labels_new=labels.view(features.shape[0],1,1).repeat(1,1,features.shape[1])
    centroids_new=torch.gather(centroids_new,1,labels_new).squeeze()
    loss=torch.nn.functional.mse_loss(centroids_new,features)
    return loss