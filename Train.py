import pickle
import glob
import awkward as ak
from time import time
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Subset
from training.gnn import GNNTrainer
import logging
from models.DynamicReductionNetwork import DynamicReductionNetwork
from torch.nn.functional import softplus
import os
from training.semiparam import get_specific_loss

def featureName(ES, coords, fracs):
    return 'features_%sES_%s_%sfrac'%(ES, coords, fracs)

def targetName(target):
    return 'targets_%s'%target

Rho_Max = 15
HoE_Max = 0.04

class Train:
    def __init__(self, folder=None, data_folder=None, weights_name = None,
            idx_name = 'all', target = 'ratioflip', ES='no', coords = 'cart', fracs='mult',
            loop = True, pool = 'max',
            in_layers = 4, agg_layers = 6, mp_layers = 3, out_layers = 2,
            hidden_dim = 64,
            device=0, train_batches = 500, valid_batch_size=5000, train_batch_size=-1,
            n_epochs=100, acc_rate = 1, 
            loss_func='dscb_loss', 
            lr_sched='Cyclic', max_lr=1e-3, min_lr=1e-7, restart_period=100, 
            gamma=1.0,
            num_classes = 6, semiparam=True, warm=None,
            latent_probe=None,
            threshold = None, epsilon=None, minalpha=None, minn=None,
            graph_features = [],
            fixedmu=None, fixedmu_target=None):
        self.folder = folder
        self.data_folder = '/home/rusack/shared/pickles/%s'%data_folder
        self.idx_name = idx_name
        self.target = target
        self.ES = ES
        self.coords = coords
        self.fracs = fracs

        self.weights_name = weights_name

        self.graph_features=graph_features

        self.loop = loop
        self.pool = pool
        self.in_layers = in_layers
        self.agg_layers = agg_layers
        self.mp_layers = mp_layers
        self.out_layers = out_layers
        self.hidden_dim = hidden_dim

        if device>=0:
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = torch.device('cpu')

        self.train_batches=train_batches
        self.train_batch_size=train_batch_size

        self.valid_batch_size=valid_batch_size

        self.acc_rate = acc_rate
        self.n_epochs = n_epochs
        self.loss_func = loss_func
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.restart_period = restart_period
        self.lr_sched = lr_sched

        self.gamma = gamma

        self.num_classes = num_classes

        self.semiparam = semiparam

        self.warm=warm

        self.latent_probe = latent_probe

        if self.target == 'logratioflip':
            self.threshold = np.log(threshold)
        else:
            self.threshold = threshold
        self.minalpha = minalpha
        self.minn = minn
        self.epsilon= epsilon

        if type(graph_features) == str:
            graph_features = [graph_features]

        self.graph_features = graph_features

        self.fixedmu = fixedmu
        self.fixedmu_target = fixedmu_target

    def load(self, predict):
        self.loadValidIdx()
        self.loadFeatures(predict)

    def loadValidIdx(self):
        prefix = '%s/%s'%(self.data_folder, self.idx_name)
        valididx_file = prefix+'_valididx.pickle'
        trainidx_file = prefix+'_trainidx.pickle'

        with open(valididx_file, "rb") as f:
            self.valid_idx = pickle.load(f)

        if os.path.exists(trainidx_file):
            with open(trainidx_file, 'rb') as f:
                self.train_idx = pickle.load(f)
        else:
            self.train_idx = np.asarray([])

        print(len(self.valid_idx), 'valid points')
        print(len(self.train_idx), 'train points')

    def loadWeights(self):
        if self.weights_name is None:
            return

        fname = '%s/%s_weights.pickle'%(self.data_folder, self.weights_name)
        with open(fname, 'rb') as f:
            self.weights = pickle.load(f)

    def loadFeatures(self, predict):
        '''
        Load in features (ie all the X y pairs)
        '''
       
        print("loading in features...")
        t0 = time()
        fname = '%s/%sfeat'%(self.data_folder, self.coords)
        if self.ES == 'yes':
            fname += '_ES'
        elif self.ES == 'scaled':
            fname += '_ES_scaled'

        data = torch.load("%s.pickle"%fname)
            
        print("\tTook %0.3f seconds"%(time()-t0))

        if len(self.graph_features)>0:
            graph_x = []        
            for var in self.graph_features:
                with open("%s/%s.pickle"%(self.data_folder, var), 'rb') as f:
                    tmp = pickle.load(f)
                    if var == 'rho':
                        tmp/=Rho_Max
                    elif var == 'Pho_HadOverEm':
                        tmp/=HoE_Max
                    graph_x.append(tmp)
            if len(graph_x)==1:
                graph_x = graph_x[0]
            else:
                graph_x = np.stack(graph_x, 1)

            print("Adding graph features to data objects...")
            for it, gx in tqdm(zip(data, graph_x), total=len(data)):
                it.graph_x = torch.from_numpy(np.asarray(gx).astype(np.float32))
            
        if not predict:
            print("loading in target...")
            t0 = time()
            with open("%s/%s_target.pickle"%(self.data_folder, self.target), 'rb') as f:
                target = pickle.load(f)
            print("\tTook %0.3f seconds"%(time()-t0))

            print("Matching targets with features...")
            for it, ta in tqdm(zip(data, target), total = len(target)):
                it.y = torch.from_numpy(np.asarray(ta).astype(np.float32))

        self.features = data

        self.loader = DataLoader(data, batch_size = self.valid_batch_size, shuffle=False, pin_memory=True)

        self.num_features = data[0].x.shape[1]
        self.datalen = len(data)

        if self.fixedmu is not None:
            print("loading in fixedmu...")
            t0 = time()
            with open(self.fixedmu, 'rb') as f:
                fixedmu = np.asarray(pickle.load(f)).astype(np.float32)

            if self.target != self.fixedmu_target:
                #then we need to map the fixedmu into the correct target space
                #ugh
                if self.fixedmu_target == 'logratioflip': #this is really the only relevant case. I might code more for completeness, but not urgently
                    factor = np.exp(-fixedmu) #this is Pred/Raw
                    if self.target == 'trueE':
                        with open("%s/rawE.pickle"%self.data_folder):
                            rawE = np.asarray(pickle.load(f))
                        fixedmu = rawE*factor
                    elif self.target == 'ratio': #true/raw
                        fixedmu = factor
                    elif self.target == 'ratioflip':
                        fixedmu = np.reciprocal(factor)
                    #there are more cases, but I don't think any of them are relevant
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            print("\tTook %0.3f seconds"%(time()-t0))

            self.fixedmu_data = fixedmu

            print("Adding fixedmu to data objects...")
            for it, fm in tqdm(zip(data, fixedmu), total=len(data)):
                it.fixedmu = torch.from_numpy(np.asarray(fm).astype(np.float32))

        print("datalen is",self.datalen)
        print("batch size is", self.loader.batch_size)
        print("ES is",self.ES,"and the number of features is", self.num_features)

    def split(self):
        train_data = Subset(self.features, self.train_idx)
        valid_data = Subset(self.features, self.valid_idx)

        if self.train_batch_size==-1:
            self.train_batch_size = int(len(train_data)/self.train_batches+0.5)

        if self.weights_name is None:
            self.train_loader = DataLoader(train_data, batch_size = self.train_batch_size,
                    shuffle=True, pin_memory=True)
            self.valid_loader = DataLoader(valid_data, batch_size = self.valid_batch_size,
                    shuffle=False, pin_memory=True)
        else:
            self.loadWeights()
            
            self.valid_weights = self.weights[self.valid_idx]
            self.train_weights = self.weights[self.train_idx]
            
            self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(self.train_weights, len(self.train_idx))
            self.valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(self.valid_weights, len(self.valid_idx))

            self.train_loader = DataLoader(train_data, batch_size = self.train_batch_size, pin_memory = True, sampler = self.train_sampler)
            self.valid_loader = DataLoader(valid_data, batch_size = self.valid_batch_size, pin_memory = True, sampler = self.valid_sampler)


    def train(self):
        weights = np.array([1.])

        trainer = GNNTrainer(output_dir = self.folder, device = self.device, 
                acc_rate = self.acc_rate)
        
        trainer.logger.setLevel(logging.DEBUG)
        strmH = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        strmH.setFormatter(formatter)
        trainer.logger.addHandler(strmH)

        trainer.build_model(name='DynamicReductionNetwork', loss_func= self.loss_func,
                optimizer='AdamW', lr_sched = self.lr_sched,
                gamma = self.gamma,
                min_lr = self.min_lr, max_lr = self.max_lr, restart_period = self.restart_period,
                input_dim = self.num_features, output_dim=self.num_classes,
                in_layers = self.in_layers, agg_layers = self.agg_layers,
                mp_layers = self.mp_layers, out_layers = self.out_layers,
                hidden_dim= self.hidden_dim,
                batch_size = self.train_batch_size, epoch_size = self.datalen, warm=self.warm, 
                threshold = self.threshold, minn = self.minn, epsilon = self.epsilon, minalpha = self.minalpha,
                graph_features = len(self.graph_features), 
                fixedmu = False if (self.fixedmu is None) else True)

        trainer.print_model_summary()
        
        self.trainSummary = trainer.train(self.train_loader, self.n_epochs, 
                valid_data_loader=self.valid_loader)


        summary_file = '%s/trainSummary.pickle'%self.folder
        with open(summary_file, 'wb') as f:
            pickle.dump(self.trainSummary, f, protocol = 4)

    def predict(self):
        '''
        Use the trained model to predict the target

        @param train: if True, do the training set. Else, do the validation set
        '''
        torch.cuda.empty_cache()

        model = DynamicReductionNetwork(input_dim=self.num_features, output_dim=self.num_classes,
                mp_layers = self.mp_layers, in_layers = self.in_layers,
                agg_layers = self.agg_layers, out_layers = self.out_layers, 
                hidden_dim = self.hidden_dim,
                loop = self.loop, pool = self.pool,
                latent_probe = self.latent_probe, graph_features = len(self.graph_features))
        model.to(self.device)

        #fname = GNNTrainer.get_model_fname(model)
        #checkpoint = '%s/checkpoints/model_checkpoint_%s.best.pth.tar' % (self.folder,fname)
        checkfolder = '%s/checkpoints'%self.folder
        checkpoint = glob.glob('%s/*.best.pth.tar'%checkfolder)[0]
        state = torch.load(checkpoint, map_location=self.device)['model']
        keys = list(state.keys())
        if keys[0].startswith('drn.'):
            model.load_state_dict(state)
        else:
            new_state = {}
            for key in keys:
                if 'edgeconv' in key:
                    splits = key.split('.')
                    rest = '.'.join(splits[1:])
                    index = int(splits[0][8:]) - 1
                    new_state[f"drn.agg_layers.{index}.{rest}"] = state[key]
                else:
                    new_state['drn.'+key] = state[key]
            model.load_state_dict(new_state)

        model.eval()

        self.y_pred = []
        if self.semiparam:
            self.sigma_pred=[]
            self.params = None
            
            semifunc, _ = get_specific_loss(self.threshold, self.minalpha, self.minn, self.epsilon)

        for data in tqdm(self.loader):
            data = data.to(self.device)
            result = model(data)
            if self.semiparam:
                if self.fixedmu is None:
                    result = semifunc(result)
                else:
                    result = semifunc(result, fixedmu = data.fixedmu)
                result = torch.stack(result)
                result = result.detach().cpu().numpy()
            else:
                result = result.detach().cpu().numpy()

            if self.latent_probe is None:
                if self.semiparam:
                    if self.params is None:
                        self.params = result
                    else:
                        self.params = np.concatenate( (self.params, result), axis=1)
                        #print(self.params.shape)
                else:
                    self.y_pred += result.tolist()
            else:
                self.y_pred += [result.cpu().detach().numpy()]

        '''
        TODO: this does not yet properly match 64-dimensional hit feature vectors with particles
        this is not a problem for latent_probe == agg_layers+1, as there is exactly 1 hit
        '''
        if self.latent_probe is not None:
            self.y_pred = ak.to_regular(ak.concatenate(self.y_pred))
            print(ak.type(self.y_pred))
            predname = '%s/latent%d.pickle'%(self.folder, self.latent_probe)
        else:
            predname = '%s/pred.pickle'%(self.folder)

        if self.semiparam:
            self.y_pred = self.params[0,:]
            self.sigma_pred = self.params[1,:]

        with open(predname, 'wb') as f:
            pickle.dump(self.y_pred, f, protocol = 4)

        if self.semiparam and self.latent_probe is None:
            sigmaname='%s/sigma_pred.pickle'%(self.folder)
            with open(sigmaname, 'wb') as f:
                pickle.dump(self.sigma_pred,f, protocol = 4)

            paramname='%s/params.pickle'%(self.folder)
            with open(paramname, 'wb') as f:
                pickle.dump(self.params, f, protocol=4)


