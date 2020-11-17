import sys
import os
import matplotlib.pyplot as plt
import argparse

class TraceIndividual():
    def __init__(self, epoch):
        self.epoch = epoch
        self.train_loss, self.valid_loss = -1.0, -1.0
        self.train_top1_acc, self.valid_top1_acc = -1.0, -1.0

    def line_to_trace(self, line):
        values = line.split()
        for i in range(1, len(values)-1):
            values[i] = values[i].strip('\'')
            values[i] = values[i].strip()
            values[i] = values[i].strip(',')
            values[i] = values[i].strip('\'')
            values[i] = float(values[i])


        self.train_loss, self.valid_loss = values[1], values[2]
        self.train_top1_acc, self.valid_top1_acc = 1.0-float(values[1]), 1.0-float(values[2])
        

class TraceAdj():
    def __init__(self, epoch):
        self.epoch = epoch
        self.train_loss, self.valid_loss = -1.0, -1.0
        self.train_big_top1_acc, self.valid_big_top1_acc = -1.0, -1.0
        self.train_small_top1_acc, self.valid_small_top1_acc = -1.0, -1.0

    def line_to_trace(self, line):
        values = line.split()
        for i in range(1, len(values)-1):
            values[i] = values[i].strip('\'')
            values[i] = values[i].strip()
            values[i] = values[i].strip(',')
            values[i] = values[i].strip('\'')
            values[i] = float(values[i])


        self.train_loss, self.valid_loss = values[1], values[4]
        self.train_big_top1_acc, self.valid_big_top1_acc = values[2], values[5]
        self.train_small_top1_acc, self.valid_small_top1_acc = values[3], values[6]

#individual_file = os.path.join(dataset_path, "individual.txt")
traces = []
individual_file_name = "/home/un270/ModelTrain/COCO/trace/individual_resnet50_60.txt"
adj_file_name = "/home/un270/ModelTrain/COCO/trace/adj_resnet50_60.txt"

def get_traces_from_file(file_name, is_combined=True):
    traces = []
    with open(file_name, 'r') as f:
        i = 1
        for line in f:
            trace = TraceAdj(i) if is_combined else TraceIndividual(i)
            line = line.strip()
            if len(line) == 0: continue

            trace.line_to_trace(line)
            i += 1
            traces.append(trace)
    return traces


def plot_traces(individual_traces, adj_trace, alphas=[4], filename="coco_accuracy_profile.png", legends=["Adjoint full", "Standard full"]):
    fig = plt.figure()
    plt.plot([trace.valid_big_top1_acc for trace in adj_trace])
    plt.plot([trace.valid_small_top1_acc for trace in adj_trace])
    plt.plot([trace.valid_top1_acc for trace in individual_traces])
    plt.ylabel('Accuracy', fontsize=16)
    fig.suptitle('Valid accuracy vs #Epochs', fontsize=20)
    plt.gca().legend(("Adjoint-{} full".format(4), "Adjoint-{} small".format(4), 'Standard full'))

    plt.xlabel('Epoch', fontsize=18)
    plt.savefig("plots/"+filename)

# Compare train vs valid loss
def plot_loss_traces(trace, is_adjoint=True, filename="loss_profile.png", legends=["Train Loss", "Valid Loss"]):
    fig = plt.figure()
    plt.plot([curr.train_loss for curr in trace])
    plt.plot([curr.valid_loss for curr in trace])
    plt.ylabel('Loss', fontsize=16)
    fig.suptitle('Train vs Valid Loss', fontsize=20)
    plt.gca().legend((legends[0], legends[1]))

    plt.xlabel('Epoch', fontsize=16)
    plt.savefig("plots/"+filename)

# Compare two adjoint trainings
def plot_adjoint_traces(indi_trace, adj1_trace, title='', filename="adj_type1_type2.png", legends=["Adjoint full", "Standard full"],compression=1):
    fig = plt.figure()
    plt.plot([trace.valid_big_top1_acc for trace in adj1_trace])
    plt.plot([trace.valid_small_top1_acc for trace in adj1_trace])
    plt.plot([trace.valid_top1_acc for trace in indi_trace])    

    plt.ylabel('Accuracy', fontsize=16)
    fig.suptitle(title, fontsize=20)
    plt.gca().legend(("Adjoint-{} full".format(4), "Adjoint-small "+compression+"X (size)", "Individual"))

    plt.xlabel('Epoch', fontsize=16)
    plt.savefig("plots/"+filename)


individual_file_name = "/home/un270/brain_mri/trace/individual.txt"
adj_3blocks_4_file = "/home/un270/brain_mri/trace/adj-3-4.txt"
adj_3blocks_2_file = "/home/un270/brain_mri/trace/adj-3-2.txt"
adj_2blocks_4_file = "/home/un270/brain_mri/trace/adj-2-4.txt"

'''
adj_file_type1_expo = "/home/un270/ModelTrain/COCO/trace/type1_loss/exponential_v2.txt"
adj_file_type2_expo = "/home/un270/ModelTrain/COCO/trace/type2_loss/exponential.txt"

adj_file_type1_x2 = "/home/un270/ModelTrain/COCO/trace/type1_loss/quadratic-4x2.txt"
adj_file_type2_x2 = "/home/un270/ModelTrain/COCO/trace/type2_loss/quadratic-4x2.txt"

adj_file_type1_min_4x2 = "/home/un270/ModelTrain/COCO/trace/type1_loss/quadratic-min-4x2.txt"
adj_file_type2_min_4x2 = "/home/un270/ModelTrain/COCO/trace/type2_loss/quadratic-min-4x2.txt"
'''

individual_traces = get_traces_from_file(individual_file_name,False)
adj_trace_3block_4 = get_traces_from_file(adj_3blocks_4_file)
adj_trace_2block_4 = get_traces_from_file(adj_3blocks_2_file)
adj_trace_3block_2 = get_traces_from_file(adj_3blocks_2_file)

plot_adjoint_traces(individual_traces,adj_trace_3block_4,title='14.7X',filename="3blocks_4.png",compression="14.7")
plot_adjoint_traces(individual_traces,adj_trace_2block_4,title='11X',filename="2blocks_4.png",compression="11")
plot_adjoint_traces(individual_traces,adj_trace_3block_2,title='3.92X',filename="3blocks_2.png",compression="3.92")


'''
import numpy as np
x = np.arange(0.0,1.0,0.01)
y1 = np.minimum(4*x**2,1)
y2 = 4*x**2
y3 = np.e**x-1

fig = plt.figure()
plt.plot([i for i in y1])
plt.plot([i for i in y2])
plt.plot([i for i in y3])
plt.ylabel('lambda(x)', fontsize=16)
plt.gca().legend(("min(4x^2,1)","4x^2","e^x-1"))
plt.xlabel('Epoch', fontsize=16)
plt.savefig("graph")
'''
