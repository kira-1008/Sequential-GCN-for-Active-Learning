import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import networkx as nx
import os
import scipy as sc
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x,k=20,y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    n = adj.shape[0]
    graph_adj_list={}
    for i in range(0,n):
         vals=[(adj[i][x],x) for x in range(0,n)]
         vals.sort()
         vals = vals[-k:]
         graph_adj_list[i]=[value[1] for value in vals] 
         adj[i]=[adj[i][x] if (adj[i][x],x) in vals else 0 for x in range(0,n)]
    adj_diag = np.sum(adj, axis=1) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj,graph_adj_list

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )

    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return  other_idx + batch

#graph central
def centralissimo(G):
    centralities = []
    #centralities.append(nx.degree_centrality(G))       #print 'degree centrality: check.'
    #centralities.append(nx.closeness_centrality(G))    #print 'closeness centrality: check.'
    #centralities.append(nx.betweenness_centrality(G))  #print 'betweenness centrality: check.'
    #centralities.append(nx.eigenvector_centrality(G))  #print 'eigenvector centrality: check.'
    centralities.append(nx.pagerank(G))                #print 'page rank: check.'
    #centralities.append(nx.harmonic_centrality(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
    	cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen

#calculate the percentage of elements smaller than the k-th element
def perc(input,k):
  idx = (input[i]<input[k] for i in range(0,len(input)))
  return np.sum(idx)/float(len(input))

#calculate the percentage of elements larger than the k-th element
def percd(input,k):
  idx = (input[i]>input[k] for i in range(0,len(input)))
  return np.sum(idx)/float(len(input))

# Select the indices of the unlablled subset, labeled_set, cycle, args data according to the methods
def query_samples(model, method, data_unlabeled,):

    if method == 'Random':
        arg = np.random.randint(SUBSET, size=SUBSET)

    if (method == 'UncertainGCN') or (method == 'CoreGCN') or (method == 'Age'):
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)
        binary_labels = torch.cat((torch.zeros([SUBSET, 1]),(torch.ones([len(labeled_set),1]))),0)

        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        print("Getting adj matrix")
        adj,adj_list = aff_to_adj(features)

        #TODO change this to get a variable argument as dataset name
        print("Started creation of networkx graph")
        directory = "res/cifar10/graphcentrality/"
        if not os.path.exists(directory):
          os.makedirs(directory)
        G = nx.Graph(adj_list)
        normcen = centralissimo(G)	#the larger the score is, the more representative the node is in the graph
        np.savetxt(directory+'normcen', normcen, fmt='%.6f', delimiter=' ', newline='\n')


        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=args.hidden_units,
                         nclass=1,
                         dropout=args.dropout_rate,nlayer=args.num_layers).cuda()
                                
        models      = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET+(cycle+1)*ADDENDUM, 1)
        nlbl = np.arange(0, SUBSET, 1)
        
        ############
        for _ in range(200):

            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss 
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

        print("Loading normality scores")
        normcen = np.loadtxt("res/cifar10"+"/graphcentrality/normcen")
        print("Successfully loaded")
        cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])
        basef=0.9
        #time sensitive parameters
        gamma = np.random.beta(1, 1.005-basef**cycle)
        alpha = beta = (1-gamma)/2

        models['gcn_module'].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
            scores, _, feat = models['gcn_module'](inputs, adj)
            
            if method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET,(SUBSET + (cycle+1)*ADDENDUM))
                sampling2 = kCenterGreedy(feat)  
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2

            elif method == "UncertainGCN":

                s_margin = args.s_margin 
                scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            elif method == "Age":
                print("Inside AGE")
                scores_subset = scores[:SUBSET].detach().cpu().numpy()
                prob = [scores_subset,1-scores_subset]
                prob = np.transpose(prob)
                entropy = sc.stats.entropy(prob)
                print(entropy[:10])
                #train_mask = sample_mask(idx_train, labels.shape[0])
    	          #entropy[train_mask+val_mask+test_mask]=-100
                entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])
                # since for us gcn is doing binary classification number of classes NCL = 2
                kmeans = KMeans(n_clusters=2, random_state=0).fit(prob)
                ed=euclidean_distances(prob,kmeans.cluster_centers_)
                ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
                edprec = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])
                finalweight = alpha*entrperc + beta*edprec + gamma*cenperc
                arg=np.argsort(finalweight)

            print("Max confidence value: ",torch.max(scores.data))
            print("Mean confidence value: ",torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[SUBSET:,0] == labels[SUBSET:,0]).sum().item() / ((cycle+1)*ADDENDUM)
            correct_unlabeled = (preds[:SUBSET,0] == labels[:SUBSET,0]).sum().item() / SUBSET
            correct = (preds[:,0] == labels[:,0]).sum().item() / (SUBSET + (cycle+1)*ADDENDUM)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)
    
    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)

        arg = get_kcg(model, ADDENDUM*(cycle+1), unlabeled_loader)

    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)        

    if method == 'VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        if args.dataset == 'fashionmnist':
            vae = VAE(28,1,3)
            discriminator = Discriminator(28)
        else:
            vae = VAE()
            discriminator = Discriminator(32)
        models      = {'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1)
        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 
    return arg
