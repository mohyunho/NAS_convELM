import torch
# from torch.autograd import Variable

class pseudoInverse(object):
    def __init__(self,params,C=1e-2,forgettingfactor=1,L =10, device="cuda"):
        
        self.params=list(params)
        self.device = device

        self.is_cuda=self.params[len(self.params)-1].is_cuda
        self.C=C
        self.L=L
        self.w=self.params[len(self.params)-1]
        # print ("self.w", self.w)
        # print ("self.w.shape", self.w.shape)
        self.w.data.fill_(0) #initialize output weight as zeros
        # For sequential learning in OS-ELM
        self.dimInput=self.params[len(self.params)-1].data.size()[1]

        # print ("self.dimInput", self.dimInput)

        self.forgettingfactor=forgettingfactor
        # self.M=Variable(torch.inverse(self.C*torch.eye(self.dimInput)),requires_grad=False, volatile=True)
        self.M=torch.inverse(self.C*torch.eye(self.dimInput))

        if self.is_cuda:
            self.M = self.M.to(self.device)

        # if self.is_cuda:
        #     self.M=self.M.cuda()

    def initialize(self):
        # self.M = Variable(torch.inverse(self.C * torch.eye(self.dimInput)),requires_grad=False, volatile=True)
        self.M = torch.inverse(self.C * torch.eye(self.dimInput))
        if self.is_cuda:
            self.M = self.M.to(self.device)
        self.w = self.params[len(self.params) - 1]
        self.w.data.fill_(0.0)

    def pseudoBig(self,inputs,targets):
        xtx = torch.mm(inputs.t(), inputs) # [ n_features * n_features ]
        dimInput=inputs.size()[1]
        I = torch.eye(dimInput)
        if self.is_cuda:
            I = I.to(self.device)
        if self.L > 0.0:
            mu = torch.mean(inputs, dim=0, keepdim=True)  # [ 1 * n_features ]
            S = inputs - mu
            S = torch.mm(S.t(), S)
            self.M = torch.inverse(xtx.data + self.C * (I.data+self.L*S.data))
        else:
            self.M = torch.inverse(xtx.data + self.C *I.data)
      

        w = torch.mm(self.M, inputs.t())
        w = torch.mm(w, targets)
        self.w.data = w.t().data
        # print ("self.w.data.shape", self.w.data.shape)


    def pseudoSmall(self,inputs,targets):
        xxt = torch.mm(inputs, inputs.t())
        numSamples=inputs.size()[0]
        # print ("numSamples", numSamples)
        I = torch.eye(numSamples)
        if self.is_cuda:
            I = I.to(self.device)
        self.M = torch.inverse(xxt.data + self.C * I.data)
        w = torch.mm(inputs.t(), self.M)
        w = torch.mm(w, targets)

        self.w.data = w.t().data
        # print ("self.w.data.shape", self.w.data.shape)

    def train(self,inputs,targets, oneHotVectorize=False):
        # print ("inputs", inputs)
        targets = targets.view(targets.size(0),-1)
        # if oneHotVectorize:
        #     targets=self.oneHotVectorize(targets=targets)
        numSamples=inputs.size()[0]

        dimInput=inputs.size()[1]
        dimTarget=targets.size()[1]



        if numSamples>dimInput:
            self.pseudoBig(inputs,targets)
        else:
            self.pseudoSmall(inputs,targets)



    # def train_sequential(self,inputs,targets):
    #     oneHotTarget = self.oneHotVectorize(targets=targets)
    #     numSamples = inputs.size()[0]
    #     dimInput = inputs.size()[1]
    #     dimTarget = oneHotTarget.size()[1]

    #     if numSamples<dimInput:
    #         I1 = torch.eye(dimInput)
    #         if self.is_cuda:
    #             I1 = I1.cuda()
    #         xtx=torch.mm(inputs.t(),inputs)
    #         self.M=torch.inverse(xtx.data+self.C*I1.data)

    #     I = torch.eye(numSamples)
    #     if self.is_cuda:
    #         I = I.cuda()

    #     self.M = (1/self.forgettingfactor) * self.M - torch.mm((1/self.forgettingfactor) * self.M,
    #                                          torch.mm(inputs.t(), torch.mm(torch.inverse(I.data + torch.mm(inputs, torch.mm((1/self.forgettingfactor)* self.M, inputs.t())).data),
    #                                          torch.mm(inputs, (1/self.forgettingfactor)* self.M))))


    #     self.w.data += torch.mm(self.M,torch.mm(inputs.t(),oneHotTarget - torch.mm(inputs,self.w.t()))).t().data


    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().data[0]+1)

        for i in range(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1

        if self.is_cuda:
            oneHotTarget=oneHotTarget.cuda()
        oneHotTarget=oneHotTarget

        return oneHotTarget

