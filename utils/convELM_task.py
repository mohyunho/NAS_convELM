#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""
import pandas as pd
from abc import abstractmethod

# from input_creator import input_gen
from utils.convELM_network import ConvElm
from utils.convELM_network import train_net
# from utils.pseudoInverse import pseudoInverse


def release_list(lst):
   del lst[:]
   del lst

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    '''
    TODO: Consider hyperparameters of ELM instead of the number of neurons in hidden layers of MLPs.
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, constant, batch, epochs, model_path, device, obj, trial):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.constant = constant
        self.batch = batch
        self.epochs = epochs
        self.model_path = model_path
        self.device = device
        self.obj = obj
        self.trial = trial

    def get_n_parameters(self):
        return 7

    def get_parameters_bounds(self):
        bounds = [
            (1, 20), #conv1_ch_mul
            (1, 20), #conv1_kernel_size
            (1, 20), #conv2_ch_mul
            (1, 20), #conv2_kernel_size
            (1, 20), #conv3_ch_mul
            (1, 20), #conv3_kernel_size
            (1, 80), #lin_mul
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        print ("######################################################################################")
        # l2_parms_lst = [1, 1e-1, 1e-2, 1e-3]
        # l2_parm = l2_parms_lst[genotype[0]-1]
        # l2_parm = self.constant
        l2_parm = 1e-3
        print("l2_params: " ,l2_parm)
        feat_len = self.train_sample_array[0].shape[1]
        win_len = self.train_sample_array[0].shape[2]
        print ("feat_len", feat_len)
        print ("win_len", win_len)
        # print ("lin_mul",  genotype[4])

        conv1_ch_mul = genotype[0]
        conv1_kernel_size = genotype[1]
        conv2_ch_mul = genotype[2]
        conv2_kernel_size = genotype[3]
        conv3_ch_mul = genotype[4]
        conv3_kernel_size = genotype[5]
        fc_mul = genotype[6]
        # lin_mul = genotype[4]

        # convELM_model = Net(feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, lin_mul, l2_parm, self.model_path)
        
        # convELM_model = Net(feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, l2_parm, self.model_path)

        convELM_model = ConvElm(feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, fc_mul, l2_parm, self.model_path, self.trial).to(self.device)

        # print("convELM_model", convELM_model)
        print(f"Model structure: {convELM_model}\n\n")

        validation = train_net(convELM_model, self.train_sample_array, self.train_label_array, self.val_sample_array,
                                    self.val_label_array, l2_parm, self.epochs, self.device)

        num_neuron = convELM_model.num_hidden_neurons

        val_value = validation[0]

        if self.obj == "soo":
            # fitness = (val_penalty,)
            fitness = (val_value,)
        elif self.obj == "nsga2":
            # fitness = (val_value, sum(num_neuron_lst))
            fitness = (val_value, num_neuron)

        print("fitness: ", fitness)

        convELM_model = None
        del convELM_model

        return fitness


