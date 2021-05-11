#!/usr/bin/env python
# coding: utf-8

# # BS6207 Project

# ## Import Libraries

# In[1]:


import random
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ## Import Pytorch Libraries

# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam


# ## Read .pdb Files

# ### Read the Proteins and Ligands .pdb Files of the Training Set

# In[3]:


def read_train_pdb(filename):
    '''
    Read a original pdb file and extract the data.
    The information of one atom is represented as a line in the array.
    Result is returned as a numpy array
    '''
    # Storing X, Y, Z coordinates of the atom and the atom type into lists
    X_list = list()
    Y_list = list()
    Z_list = list()
    atom_list = list()
    
    with open(filename, 'r') as file:
        for strline in file.readlines():            
            # Removes all the white spaces
            stripped_line = strline.strip()           
            # Error catch
            line_length = len(stripped_line)            
            if line_length < 78:
                print("ERROR: line length is different. Expected>=78, current={}".format(line_length))
                break               
            else:
                # Append to list
                X_list.append(float(stripped_line[30:38].strip()))
                Y_list.append(float(stripped_line[38:46].strip()))
                Z_list.append(float(stripped_line[46:54].strip()))
                # h: hydrophobic; p: polar
                atom_list.append(
                    'h' if stripped_line[76:78].strip() == 'C' else 'p',
                )
        
    return X_list, Y_list, Z_list, atom_list


# ### Read the Proteins and Ligands .pdb Files of the Testing Set

# In[4]:


def read_test_pdb(filename):
    '''
    Read a original pdb file and extract the data.
    The information of one atom is represented as a line in the array.
    Result is returned as a numpy array
    '''
    # Storing X, Y, Z coordinates of the atom and the atom type into lists
    X_list = list()
    Y_list = list()
    Z_list = list()
    atom_list = list()
    
    with open(filename, 'r') as file:
        for strline in file.readlines():    
            # Removes all the white spaces
            stripped_line = strline.strip()
            # Split a tabs
            splitted_line = stripped_line.split('\t')            
            # Append to list
            X_list.append(float(splitted_line[0])),
            Y_list.append(float(splitted_line[1])),
            Z_list.append(float(splitted_line[2])),
            atom_list.append(str(splitted_line[3]))
            
    return X_list, Y_list, Z_list, atom_list


# ## Load Data into Dictionary
# - training_data (3000 samples of protein-ligand pairs)
# - testing_data (824 samples of protein-ligand pairs)

# ### Training Data

# In[5]:


# Load training data into dictionary
training_data = {
    'proteins': list(),
    'ligands': list()
}

for i in range(1,3001):
    training_data['proteins'].append(
        # Load your own file path for the training data, include training_data/{:04d}_pro_cg.pdb behind
        read_train_pdb('/Users/isaaclzh/OneDrive/BS6207/BS6207 Project/training_data/{:04d}_pro_cg.pdb'.format(i)))
    training_data['ligands'].append(
        # Load your own file path for the training data, include training_data/{:04d}_lig_cg.pdb behind
        read_train_pdb('/Users/isaaclzh/OneDrive/BS6207/BS6207 Project/training_data/{:04d}_lig_cg.pdb'.format(i)))


# ### Testing Data

# In[6]:


# Load testing data into dictionary
testing_data = {
    'proteins': list(),
    'ligands': list()
}

for i in range(1,825):
    testing_data['proteins'].append(
        # Load your own file path for the testing data, same as training_data
        read_test_pdb('/Users/isaaclzh/OneDrive/BS6207/BS6207 Project/testing_data_release/testing_data/{:04d}_pro_cg.pdb'.format(i)))
    testing_data['ligands'].append(
        # Load your own file path for the testing data, same as training_data
        read_test_pdb('/Users/isaaclzh/OneDrive/BS6207/BS6207 Project/testing_data_release/testing_data/{:04d}_lig_cg.pdb'.format(i)))


# ## Visualization of Protein-Ligand Pairing
# - Protein molecule is in orange
# - Ligand molecule is in blue

# In[7]:


# Define plot function
def plot3d(protein,ligand):
    '''
    Plots a 3D diagram of protein-ligand pairing
    '''  
    plt.rcParams['figure.figsize'] = (14,16)
    fig = plt.figure() 
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(protein[0], protein[1], protein[2], c='darkorange', marker='o')
    ax.scatter(ligand[0], ligand[1], ligand[2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# ### Positive-Pairing Example
# - Take for instance 0004_pro_cg.pdb and 0004_lig_cg.pdb with positive pairing

# In[8]:


print('{:04d}_pro_cg.pdb, {:04d}_lig_cg.pdb'.format(4, 4))
plot3d(training_data['proteins'][3],training_data['ligands'][3])


# ### Negative-Pairing Example
# - Take for instance 0004_pro_cg.pdb and 0006_lig_cg.pdb with negative pairing

# In[9]:


print('{:04d}_pro_cg.pdb, {:04d}_lig_cg.pdb'.format(4, 6))
plot3d(training_data['proteins'][3],training_data['ligands'][5])


# - In total there are 3000 positive-pairing examples and 2999^2 negative-pairing examples in the training data

# ## Normalize Protein Around Centroid of Ligand
# - normalized_training_data (3000 samples of protein-ligand pairs)
# - normalized_testing_data (824 samples of protein-ligand pairs)

# ### Normalized Training Data

# In[10]:


# Load normalized training data into dictionary
normalized_training_data = {
    'proteins': list(),
    'ligands': list()
}

for i in range(len(training_data['proteins'])):
    lig_centroid = np.expand_dims(np.mean(training_data['ligands'][i][:3], axis = 1), axis = 1)
    prot_ = (np.array(np.round((training_data['proteins'][i][:3] - lig_centroid), decimals = 3))).tolist()
    prot_.append(training_data['proteins'][i][3])
    normalized_training_data['proteins'].append(prot_)
    lig_ = (np.array(np.round((training_data['ligands'][i][:3] - lig_centroid), decimals = 3))).tolist()
    lig_.append(training_data['ligands'][i][3])
    normalized_training_data['ligands'].append(lig_)


# ### Normalized Testing Data

# In[11]:


# Load normalized testing data into dictionary
normalized_testing_data = {
    'proteins': list(),
    'ligands': list()
}

for i in range(len(testing_data['proteins'])):
    lig_centroid = np.expand_dims(np.mean(testing_data['ligands'][i][:3], axis = 1), axis = 1)
    prot_ = (np.array(np.round((testing_data['proteins'][i][:3] - lig_centroid), decimals = 3))).tolist()
    prot_.append(testing_data['proteins'][i][3])
    normalized_testing_data['proteins'].append(prot_)
    lig_ = (np.array(np.round((testing_data['ligands'][i][:3] - lig_centroid), decimals = 3))).tolist()
    lig_.append(testing_data['ligands'][i][3])
    normalized_testing_data['ligands'].append(lig_)


# ## Visualization of Normalized Protein-Ligand Pairing

# ### Positive-Pairing Example
# - Take for instance 0004_pro_cg.pdb and 0004_lig_cg.pdb with positive pairing

# In[12]:


print('{:04d}_pro_cg.pdb, {:04d}_lig_cg.pdb'.format(4, 4))
plot3d(normalized_training_data['proteins'][3],normalized_training_data['ligands'][3])


# ## Global Minimum Across Normalized X, Y, Z Coordinates
# - Shift coordinates by at least 30 so that protein-ligand can remain in the positive axis

# In[13]:


# Average global minimum coordinates
min_X = 0
min_Y = 0
min_Z = 0

min_coordinates = list()
for i in range(len(normalized_training_data['proteins'])):
    min_coordinates.append([min(normalized_training_data['proteins'][i][0]),
                            min(normalized_training_data['proteins'][i][1]),
                            min(normalized_training_data['proteins'][i][2])])

for i in range(len(normalized_training_data['proteins'])):
    min_X += min_coordinates[i][0]
    min_Y += min_coordinates[i][1]
    min_Z += min_coordinates[i][2]
    
print('The average minimum coordinates for X, Y, Z is [{}, {}, {}]'.format(round(min_X/len(normalized_training_data['proteins']), 2),
                                                                        round(min_Y/len(normalized_training_data['proteins']), 2),
                                                                        round(min_Z/len(normalized_training_data['proteins']), 2)))


# ## Absolute Maximum Across Normalized X, Y, Z Coordinates
# - Cube size could be set to 48 x 48 x 48 to account for at least 75% of the atoms inside protein-ligand pair

# In[14]:


# Record the maximum size of each axis to estimate the size of cube
x_max = list()
y_max = list()
z_max = list()

for i in range(len(normalized_training_data['proteins'])):
    x_max.append(max(np.abs(normalized_training_data['proteins'][i][0])))
    y_max.append(max(np.abs(normalized_training_data['proteins'][i][1])))
    z_max.append(max(np.abs(normalized_training_data['proteins'][i][2])))


# In[15]:


max_size = pd.DataFrame({'max_X':x_max,'max_Y':y_max,'max_Z':z_max})
max_size.describe()


# - I realized the model cannot train on 48 x 48 x 48, so I resized it to 25 x 25 x 25 and scale the molecules by a factor of 4 when generating positive and negative pairings

# ## Split Normalized Training Data into 80% Training and 20% Validation
# - normalized_training_split_data (2400 samples of protein-ligand pairs)
# - normalized_validation_split_data (600 samples of protein-ligand pairs)

# In[16]:


# Split training data into 80% training and 20% validation
n = int(len(normalized_training_data['proteins'])*0.8)

normalized_training_split_data = {
    'proteins': normalized_training_data['proteins'][:n],
    'ligands': normalized_training_data['ligands'][:n]    
}

normalized_validation_split_data = {
    'proteins': normalized_training_data['proteins'][n:],
    'ligands': normalized_training_data['ligands'][n:]
}


# - Adjust the train-test split percentage rate when tuning parameters

# ## Generate Samples
# - Cube is a 3D structure 25 x 25 x 25 pixels
# - 2 channels input
# - First channel (molecules): -1 for protein; 1 for ligand
# - Second channel (atom type): -1 for polar; 1 for hydrophobic

# ### Positive-Pairing Samples

# In[17]:


def generate_positive_pairing(cube_size, normalized_training_data):
    
    pos_cubes_ = []
    scaling = 100//cube_size
    phase_shift = (cube_size - 1 ) // 2
    
    for i in range(int(len(normalized_training_data['proteins']))):
        pos_cubes = torch.zeros(2,cube_size,cube_size,cube_size)
        
        # Protein
        for j in range(len(normalized_training_data['proteins'][i][0])):
            x_p, y_p, z_p = normalized_training_data['proteins'][i][0][j], normalized_training_data['proteins'][i][1][j], normalized_training_data['proteins'][i][2][j]
            x_p, y_p, z_p = int(round(x_p/scaling + phase_shift)), int(round(y_p/scaling + phase_shift)), int(round(z_p/scaling + phase_shift))
            
            # Pruning
            if x_p >= cube_size or y_p >= cube_size or z_p >= cube_size or x_p < 0 or y_p < 0 or z_p < 0 :
                continue
            if pos_cubes[0,x_p,y_p,z_p] !=0 or pos_cubes[1,x_p,y_p,z_p] !=0:
                continue 
                
            # First channel
            pos_cubes[0,x_p,y_p,z_p] = -1
            
            # Second channel
            if normalized_training_data['proteins'][i][3][j] == 'h':
                pos_cubes[1,x_p,y_p,z_p] = 1
            else:
                pos_cubes[1,x_p,y_p,z_p] = -1
                
        # Ligands 
        for k in range(len(normalized_training_data['ligands'][i][0])):
            x_l, y_l, z_l = normalized_training_data['ligands'][i][0][k], normalized_training_data['ligands'][i][1][k], normalized_training_data['ligands'][i][2][k]
            x_l, y_l, z_l = int(round(x_l/scaling + phase_shift)), int(round(y_l/scaling + phase_shift)), int(round(z_l/scaling + phase_shift))
            
            # Pruning
            if x_l > cube_size or y_l > cube_size or z_l > cube_size or x_l < 0 or y_l < 0 or z_l < 0 :
                continue
            if pos_cubes[0,x_l,y_l,z_l] !=0 or pos_cubes[1,x_l,y_l,z_l] !=0:
                continue
                
            # First channel
            pos_cubes[0,x_l,y_l,z_l] = 1
            
            # Second channel
            if normalized_training_data['ligands'][i][3][k] == 'h':
                pos_cubes[1,x_l,y_l,z_l] = 1
            else:
                pos_cubes[1,x_l,y_l,z_l] = -1
                
        pos_cubes_.append(pos_cubes)
                
    return pos_cubes_


# In[18]:


# Generate positive pairings for train and validation data
pos_training_cubes = torch.stack(generate_positive_pairing(25, normalized_training_split_data))
pos_validation_cubes = torch.stack(generate_positive_pairing(25, normalized_validation_split_data))


# In[19]:


print('Shape of pos_training_cubes:',pos_training_cubes.shape)
print('Shape of pos_validation_cubes:',pos_validation_cubes.shape)


# ### Negative-Pairing Samples (1-to-1)
# - 1 protein to 1 possible combination of negative pairings

# In[20]:


def generate_negative_pairing_onetoone(cube_size, normalized_training_data):
    
    neg_cubes_ = []
    scaling = 100//cube_size
    phase_shift = (cube_size - 1 ) // 2
    
    for i in range(int(len(normalized_training_data['proteins']))):
        neg_cubes = torch.zeros(2,cube_size,cube_size,cube_size)
        
        # Protein
        for j in range(len(normalized_training_data['proteins'][i][0])):
            x_p, y_p, z_p = normalized_training_data['proteins'][i][0][j], normalized_training_data['proteins'][i][1][j], normalized_training_data['proteins'][i][2][j]
            x_p, y_p, z_p = int(round(x_p/scaling + phase_shift)), int(round(y_p/scaling + phase_shift)), int(round(z_p/scaling + phase_shift))
            
            # Pruning
            if x_p >= cube_size or y_p >= cube_size or z_p >= cube_size or x_p < 0 or y_p < 0 or z_p < 0 :
                continue
            if neg_cubes[0,x_p,y_p,z_p] !=0 or neg_cubes[1,x_p,y_p,z_p] !=0:
                continue 
                
            # First channel
            neg_cubes[0,x_p,y_p,z_p] = -1
            
            # Second channel
            if normalized_training_data['proteins'][i][3][j] == 'h':
                neg_cubes[1,x_p,y_p,z_p] = 1
            else:
                neg_cubes[1,x_p,y_p,z_p] = -1
                
        # Ligand
        choice_ = random.choice(list(range(i)) + list(range(i+1, int(len(normalized_training_data['proteins'])))))        
        for k in range(len(normalized_training_data['ligands'][choice_][0])):
            x_l, y_l, z_l = normalized_training_data['ligands'][choice_][0][k], normalized_training_data['ligands'][choice_][1][k], normalized_training_data['ligands'][choice_][2][k]
            x_l, y_l, z_l = int(round(x_l/scaling + phase_shift)), int(round(y_l/scaling + phase_shift)), int(round(z_l/scaling + phase_shift))
            
            # Pruning
            if x_l > cube_size or y_l > cube_size or z_l > cube_size or x_l < 0 or y_l < 0 or z_l < 0 :
                continue
            if neg_cubes[0,x_l,y_l,z_l] !=0 or neg_cubes[1,x_l,y_l,z_l] !=0:
                continue
                
            # First channel
            neg_cubes[0,x_l,y_l,z_l] = 1
            
            # Second channel
            if normalized_training_data['ligands'][choice_][3][k] == 'h':
                neg_cubes[1,x_l,y_l,z_l] = 1
            else:
                neg_cubes[1,x_l,y_l,z_l] = -1
                
        neg_cubes_.append(neg_cubes)        
             
    return neg_cubes_


# In[21]:


# Generate negative pairings for train and validation data
neg_training_cubes_ = torch.stack(generate_negative_pairing_onetoone(25, normalized_training_split_data))
neg_validation_cubes_ = torch.stack(generate_negative_pairing_onetoone(25, normalized_validation_split_data))


# In[22]:


print('Shape of neg_training_cubes_:', neg_training_cubes_.shape)
print('Shape of neg_validation_cubes_:', neg_validation_cubes_.shape)


# ### Negative-Pairing Samples (1-to-100)
# - 1 protein to 100 possible combinations of negative pairings

# In[23]:


def generate_negative_pairing(cube_size, normalized_training_data):
    
    neg_cubes_ = []
    scaling = 100//cube_size
    phase_shift = (cube_size - 1 ) // 2

    for i in range(int(len(normalized_training_data['proteins']))):
        neg_cubes = torch.zeros(2,cube_size,cube_size,cube_size)
        check_ = []
        counter_ = 1
        
        # Protein
        for j in range(len(normalized_training_data['proteins'][i][0])):
            x_p, y_p, z_p = normalized_training_data['proteins'][i][0][j], normalized_training_data['proteins'][i][1][j], normalized_training_data['proteins'][i][2][j]
            x_p, y_p, z_p = int(round(x_p/scaling + phase_shift)), int(round(y_p/scaling + phase_shift)), int(round(z_p/scaling + phase_shift))

            # Pruning 
            if x_p >= cube_size or y_p >= cube_size or z_p >= cube_size or x_p < 0 or y_p < 0 or z_p < 0 :
                continue
            if neg_cubes[0,x_p,y_p,z_p] !=0 or neg_cubes[1,x_p,y_p,z_p] !=0:
                continue
                
            # First channel
            neg_cubes[0,x_p,y_p,z_p] = -1
            
            # Second channel
            if normalized_training_data['proteins'][i][3][j] == 'h':
                neg_cubes[1,x_p,y_p,z_p] = 1
            else:
                neg_cubes[1,x_p,y_p,z_p] = -1
                
        # Ligands
        # 100 different ligand combinations
        while counter_ <= 100:                   
            choice_ = random.choice(list(range(i)) + list(range(i+1, int(len(normalized_training_data['proteins'])))))

            if choice_ not in check_:
                counter_ += 1
                check_.append(choice_)
                
                for k in range(len(normalized_training_data['ligands'][choice_][0])):
                    x_l, y_l, z_l = normalized_training_data['ligands'][choice_][0][k], normalized_training_data['ligands'][choice_][1][k], normalized_training_data['ligands'][choice_][2][k]
                    x_l, y_l, z_l = int(round(x_l/scaling + phase_shift)), int(round(y_l/scaling + phase_shift)), int(round(z_l/scaling + phase_shift))

                    # Pruning
                    if x_l > cube_size or y_l > cube_size or z_l > cube_size or x_l < 0 or y_l < 0 or z_l < 0:
                        continue
                    if neg_cubes[0,x_l,y_l,z_l] !=0 or neg_cubes[1,x_l,y_l,z_l] !=0:
                        continue
                        
                    # First channel
                    neg_cubes[0,x_l,y_l,z_l] = 1
            
                    # Second channel
                    if normalized_training_data['ligands'][choice_][3][k] == 'h':
                        neg_cubes[1,x_l,y_l,z_l] = 1
                    else:
                        neg_cubes[1,x_l,y_l,z_l] = -1 
                
        neg_cubes_.append(neg_cubes)         
       
    return neg_cubes_


# In[24]:


# Generate negative pairings for train and validation data
neg_training_cubes = torch.stack(generate_negative_pairing(25, normalized_training_split_data))
neg_validation_cubes = torch.stack(generate_negative_pairing(25, normalized_validation_split_data))


# In[25]:


print('Shape of neg_training_cubes:', neg_training_cubes.shape)
print('Shape of neg_validation_cubes:', neg_validation_cubes.shape)


# - Adjust cube_size, phase_shift and scaling when tuning paramters

# ## Concatenate Cubes

# ### Training
# - pos_training_cubes
# - neg_training_cubes_ (1-to-1)
# - neg_training_cubes (1-to-100)

# In[26]:


# Concatenate training cubes (1-to-1)
training_data_ = torch.cat((pos_training_cubes,neg_training_cubes_),0)
# Create training labels, 1 is bind; 0 is no bind
training_labels_ = torch.tensor([1]*2400 + [0]*2400)


# In[27]:


# Concatenate training cubes (1-to-100)
training_data = torch.cat((pos_training_cubes,neg_training_cubes),0)
# Create training labels, 1 is bind; 0 is no bind
training_labels = torch.tensor([1]*2400 + [0]*2400)


# In[28]:


print('Shape of training_data_:',training_data_.shape)
print('Shape of training_labels_:',training_labels_.shape)


# In[29]:


print('Shape of training_data:',training_data.shape)
print('Shape of training_labels:',training_labels.shape)


# - Total 4800 positive & negative training data 
# - Total 4800 training labels

# In[30]:


train_dataset_ = torch.utils.data.TensorDataset(training_data_, training_labels_)
train_loader_ = torch.utils.data.DataLoader(train_dataset_, batch_size=64, shuffle=True)


# In[31]:


train_dataset = torch.utils.data.TensorDataset(training_data, training_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# ### Validation 
# - pos_validation_cubes
# - neg_validation_cubes (1-to-1)
# - neg_validation_cubes (1-to-100)

# In[32]:


# Concatenate validation cubes (1-to-1)
validation_data_ = torch.cat((pos_validation_cubes,neg_validation_cubes_),0)
# Create validation labels, 1 is bind; 0 is no bind
validation_labels_ = torch.tensor([1]*600 + [0]*600)


# In[33]:


# Concatenate validation cubes (1-to-100)
validation_data = torch.cat((pos_validation_cubes,neg_validation_cubes),0)
# Create validation labels, 1 is bind; 0 is no bind
validation_labels = torch.tensor([1]*600 + [0]*600)


# In[34]:


print('Shape of validation_data_:',validation_data_.shape)
print('Shape of validation_labels_:',validation_labels_.shape)


# In[35]:


print('Shape of validation_data:',validation_data.shape)
print('Shape of validation_labels:',validation_labels.shape)


# - Total 1200 positive & negative validation data in validation_data
# - Total 1200 validation labels in validation_labels

# - Batch size is set to 64

# In[36]:


validation_dataset_ = torch.utils.data.TensorDataset(validation_data_, validation_labels_)
validation_loader_ = torch.utils.data.DataLoader(validation_dataset_, batch_size=64, shuffle=True)


# In[37]:


validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)


# ## Define CNN Model 

# In[38]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # The first convolutional layer takes in 2 input channels, and generates 16 output channels
        self.conv1 = nn.Conv3d(2,16,kernel_size=3,stride=1,padding=0)
        
        # The second convolutional layer takes in 16 input channels, and generates 32 output channels
        self.conv2 = nn.Conv3d(16, 32,kernel_size=3,stride=1, padding=0)
        
        # The third convolutional layer takes in 32 input channels, and generates 64 output channels
        self.conv3 = nn.Conv3d(32, 64,kernel_size=3,stride=1, padding=0)
        
        # Max pooling with kernel size of 2x2
        self.pool = nn.MaxPool3d(2, 2)
        
        # A drop layer deletes 30% of the features to help prevent overfitting
        self.dropout = nn.Dropout3d(0.3)
        
        # Batch normalizations
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.batchnorm2 = nn.BatchNorm3d(32)
        self.batchnorm3 = nn.BatchNorm3d(64)
        
        # Flatten layers
        self.fc1 = nn.Linear(64*1*1*1,1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.batchnorm1(self.pool(F.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool(F.relu(self.conv2(x))))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten layer
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x),dim = 1)
        return x


# ## Create an Instance of the Model

# - model_ is for 1-to-1 pairing

# In[39]:


model_ = CNN()
print(model_)


# - model is for 1-to-100 pairing

# In[40]:


model = CNN()
print(model)


# ## Training Function

# In[41]:


def train(model, train_loader, optimizer, epoch): 
    
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    
    # Process in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Set batch
        batch = batch_idx + 1
        
        # Reset the optimizer
        optimizer.zero_grad()     
        
        # Push the data forward through the model layers
        output = model(data)       
        
        # Get the loss
        loss = loss_criteria(output, target)
        
        # Keep a running total
        train_loss += loss.item()       
        
        # Backpropagate
        loss.backward()
        optimizer.step()     
        
        # Print metrics 
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # Return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


# ## Validation Function

# In[42]:


def validation(model, validation_loader):
    
    # Switch the model to evaluation mode 
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        batch_count = 0
        
        for data, target in validation_loader:
            batch_count += 1         
            
            # Get the predicted classes for this batch
            output = model(data)    
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    validation_predict = 100. * correct / len(validation_loader.dataset)
    
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, correct, len(validation_loader.dataset),
        validation_predict))
    
    # Return average loss for the epoch
    return avg_loss, validation_predict


# ## Training the Model

# - Batch size is set to 64

# In[43]:


# Set random seed
torch.manual_seed(1)

# Use an "Adam" optimizer to adjust weights
optimizer_ = optim.Adam(model_.parameters(), lr=0.001)
# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums_ = []
training_loss_ = []
validation_loss_ = []
validation_predict_ = []

# Train over 20 epochs
epochs = 20
for epoch in range(1, epochs + 1):
    train_loss = train(model_, train_loader_, optimizer_, epoch)
    test_loss, valid_predict = validation(model_, validation_loader_)
    epoch_nums_.append(epoch)
    training_loss_.append(train_loss)
    validation_loss_.append(test_loss)
    validation_predict_.append(valid_predict)


# In[44]:


# Set random seed
torch.manual_seed(1)

# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []
validation_predict = []

# Train over 20 epochs
epochs = 20
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch)
    test_loss, valid_predict = validation(model, validation_loader)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    validation_predict.append(valid_predict)


# ## Plot of Training & Validation Accuracy

# ### 1-to-1

# In[45]:


plt.figure(figsize=(14,10))
plt.plot(epoch_nums_, training_loss_)
plt.plot(epoch_nums_, validation_loss_)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(['Training', 'Validation'], loc='upper right', prop={"size":14})
plt.title('Plot of Training and Validation Loss wrt Number of Epochs (1-to-1)', fontsize=22)
plt.show()


# In[48]:


import matplotlib.transforms as transforms
import matplotlib.ticker as mtick

fig, ax=plt.subplots()
ax.plot(epoch_nums_, validation_predict_)
ax.axhline(y=57.40,linestyle='--',color='red',xmax=0.71)
plt.title('Plot of Accuracy wrt Number of Epochs (1-to-1)', fontsize=22)
fig.set_size_inches(14, 10)
trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(0,57.4, "{:.1f}{}".format(57.4, '%'), color="red", transform=trans, 
        ha="right", va="center")

ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('Accuracy', fontsize=20)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# ### 1-to-100

# In[49]:


plt.figure(figsize=(14,10))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(['Training', 'Validation'], loc='upper right', prop={"size":14})
plt.title('Plot of Training and Validation Loss wrt Number of Epochs (1-to-100)', fontsize=22)
plt.show()


# In[50]:


fig, ax=plt.subplots()
ax.plot(epoch_nums, validation_predict)
plt.title('Plot of Accuracy wrt Number of Epochs (1-to-100)', fontsize=22)
fig.set_size_inches(14, 10)
trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('Accuracy', fontsize=20)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# ## Plot of Confusion Matrix

# ### 1-to-1

# In[51]:


truelabels_ = []
predictions_ = []

for data, target in validation_loader_:
    for label in target.data.numpy():
        truelabels_.append(label)
    for prediction in model_(data):
        predictions_.append(torch.argmax(prediction).item())
        
cm_ = confusion_matrix(truelabels_, predictions_)


# In[52]:


df_cm_ = pd.DataFrame(cm_, index = ['No Bind', 'Bind'], columns = ['No Bind', 'Bind'])
plt.figure(figsize = (14,12))
sns.heatmap(df_cm_, annot=True, annot_kws={"size": 40}, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Shape", fontsize = 20)
plt.ylabel("True Shape", fontsize = 20)
plt.title('Prediction on validation set (1-to-1)', fontsize=22)
plt.show()


# ### 1-to-100

# In[53]:


truelabels = []
predictions = []

for data, target in validation_loader:
    for label in target.data.numpy():
        truelabels.append(label)
    for prediction in model(data):
        predictions.append(torch.argmax(prediction).item())
        
cm = confusion_matrix(truelabels, predictions)


# In[54]:


df_cm = pd.DataFrame(cm, index = ['No Bind', 'Bind'], columns = ['No Bind', 'Bind'])
plt.figure(figsize = (14,12))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 40}, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Shape", fontsize = 20)
plt.ylabel("True Shape", fontsize = 20)
plt.title('Prediction on validation set (1-to-100)', fontsize=22)
plt.show()


# ## Test on normalized_testing_data

# In[55]:


def generate_testing_pairing(cube_size, normalized_training_data):
    
    test_cubes = []
    scaling = 100//cube_size
    phase_shift = (cube_size - 1 ) // 2

    for i in range(int(len(normalized_training_data['proteins']))):
        test_cubes_ = torch.zeros(2,cube_size,cube_size,cube_size)
        
        # Protein
        for j in range(len(normalized_training_data['proteins'][i][0])):
            x_p, y_p, z_p = normalized_training_data['proteins'][i][0][j], normalized_training_data['proteins'][i][1][j], normalized_training_data['proteins'][i][2][j]
            x_p, y_p, z_p = int(round(x_p/scaling + phase_shift)), int(round(y_p/scaling + phase_shift)), int(round(z_p/scaling + phase_shift))

            # Pruning 
            if x_p >= cube_size or y_p >= cube_size or z_p >= cube_size or x_p < 0 or y_p < 0 or z_p < 0 :
                continue
            if test_cubes_[0,x_p,y_p,z_p] !=0 or test_cubes_[1,x_p,y_p,z_p] !=0:
                continue
                
            # First channel
            test_cubes_[0,x_p,y_p,z_p] = -1
            
            # Second channel
            if normalized_training_data['proteins'][i][3][j] == 'h':
                test_cubes_[1,x_p,y_p,z_p] = 1
            else:
                test_cubes_[1,x_p,y_p,z_p] = -1
                
        test_cubes.append(test_cubes_)  
        
    return test_cubes


# In[56]:


# Generate protein cubes for testing data
testing_cubes = generate_testing_pairing(25, normalized_testing_data)
testing_tensor = torch.stack(testing_cubes)
print('Shape of testing_cubes:', testing_cubes[0].shape)
print('Shape of testing_tensor:', testing_tensor.shape)


# In[57]:


def match_with_ligand(cube_size, normalized_training_data, protein_cube):
    
    matched_cube = []
    scaling = 100//cube_size
    phase_shift = (cube_size - 1 ) // 2

    for i in range(int(len(normalized_training_data['ligands']))):
        
        # Ligand
        for k in range(len(normalized_training_data['ligands'][i][0])):
            x_l, y_l, z_l = normalized_training_data['ligands'][i][0][k], normalized_training_data['ligands'][i][1][k], normalized_training_data['ligands'][i][2][k]
            x_l, y_l, z_l = int(round(x_l/scaling + phase_shift)), int(round(y_l/scaling + phase_shift)), int(round(z_l/scaling + phase_shift))
        
            # Pruning
            if x_l > cube_size or  y_l > cube_size or z_l > cube_size or x_l < 0 or y_l < 0 or z_l < 0:        
                continue
            if protein_cube[0,x_l,y_l,z_l] !=0 or protein_cube[1,x_l,y_l,z_l] !=0:
                continue
                
            # First channel
            protein_cube[0,x_l,y_l,z_l] = 1
            
            # Second channel
            if normalized_training_data['ligands'][i][3][k] == 'h':
                protein_cube[1,x_l,y_l,z_l] = 1
            else:
                protein_cube[1,x_l,y_l,z_l] = -1

        matched_cube.append(protein_cube)
        
    return matched_cube


# ### Test on 1 protein with 824 matched ligands

# In[64]:


matched_cube = match_with_ligand(25, normalized_testing_data, testing_cubes[1])
matched_cube_ = torch.stack(matched_cube)
print('Shape of matched_cube_:', matched_cube_.shape)


# - Test with 1-to-100 model

# In[69]:


out = model(matched_cube_)
prob, pred = torch.max(out, dim=1)
max_prob = max(prob)


# In[77]:


pred


# ### There is an error in my prediction for the test data set to predict which ligand binds to that particular protein

# In[ ]:


prob_test = []
pred_test = []
for i in range(824):
    matched_cube__ = match_with_ligand(25, normalized_testing_data, testing_cubes[i])
    matched_cube__ = torch.stack(matched_cube__)
    out__ = model_(matched_cube__)
    prob__,pred__ = torch.max(out__, dim=1)

