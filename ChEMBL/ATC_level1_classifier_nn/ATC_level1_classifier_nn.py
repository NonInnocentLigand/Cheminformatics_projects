from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale, MultiLabelBinarizer
from sklearn.utils import shuffle
from tqdm import tqdm

torch.manual_seed(42)

# load the selected data from the PotgreSQL command to the chembl_35 database
atc_level5_df = pd.read_csv('ATC_ChEMBL_35.csv')

# Remove repeats classifications on level1
atc_level5_df.drop_duplicates(["molregno", "level1"], inplace=True)

# encode level1 categories as one hot encoded array
mlb = MultiLabelBinarizer()
level1_arr = mlb.fit_transform(atc_level5_df['level1'])
atc_level5_df['level1_arr'] = list(mlb.fit_transform(atc_level5_df['level1']))


def get_multiclass(list_of_bit_vectors):
    array_len = len(list_of_bit_vectors[0])
    multi_vector = np.zeros((array_len,), dtype=int)
    for bit_v in list_of_bit_vectors:
        multi_vector = np.add(multi_vector, np.asarray(bit_v))
    return multi_vector


multiclass_dict = {}
for molregno in atc_level5_df['molregno'].unique():
    level1_values = atc_level5_df[atc_level5_df['molregno'] == molregno]
    level1_index = list(level1_values.index)[0]
    multiclass = get_multiclass(level1_values['level1_arr'].to_list())
    multiclass_dict[level1_index] = multiclass

multiclass_series = pd.Series(multiclass_dict, name='multiclass')
atc_level5_df = pd.merge(atc_level5_df, multiclass_series, left_index=True, right_index=True)

# Shuffle the data
atc_level5_df = shuffle(atc_level5_df)

# generate fingerprints (i.e. bit vectors encoding molecular structure information) column for dataframe using RDKit


def get_fingerprint_from_smiles(smiles_string: str):
    fpgen = AllChem.GetRDKitFPGenerator()
    fp = fpgen.GetFingerprint(Chem.MolFromSmiles(smiles_string))
    arr = np.zeros((len(fp),), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


fingerprints = [get_fingerprint_from_smiles(x) for x in atc_level5_df['canonical_smiles']]
atc_level5_df['fingerprints'] = fingerprints

# Splitting up shuffled data into training, validation, and test sets using 70:15:15 ratio
# for each label in level1
atc_level1_values = atc_level5_df['level1'].unique()  # these are the tarted classifications y_hat for training/testing
training_df = pd.DataFrame(columns=list(atc_level5_df.keys()))
validation_df = training_df.copy()
testing_df = training_df.copy()

for level1_value in atc_level1_values:
    df = atc_level5_df[atc_level5_df['level1'] == level1_value].copy()
    number_of_values = df.shape[0]
    iloc_slice_1 = int(round(number_of_values*0.7, 0) - 1)
    iloc_slice_2 = int(round((number_of_values - iloc_slice_1) / 2, 0) + iloc_slice_1)
    iloc_slice_3 = int(iloc_slice_2 + 1)
    training_df = pd.concat([training_df, df.iloc[0:iloc_slice_1]])
    validation_df = pd.concat([validation_df, df.iloc[iloc_slice_1+1:iloc_slice_2]])
    testing_df = pd.concat([testing_df,  df.iloc[iloc_slice_3:number_of_values]])

# define the features (in this case molecular fingerprints) and the targets (categories from level1)
features = 'fingerprints'
target = 'level1_arr'

# shuffle again because somehow it becomes unshuffled
training_df = shuffle(training_df)
validation_df = shuffle(validation_df)
testing_df = shuffle(testing_df)

# creating torch Dataset objects


class ATCDataSet(Dataset):
    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = dataframe[features]
        self.target = dataframe[target]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        features = torch.tensor(self.features.iloc[index], dtype=torch.float32)
        target = torch.tensor(self.target.iloc[index], dtype=torch.float32)  # make sure this is right d_type for loss fn
        return features, target


train_dataset = ATCDataSet(training_df, features, target)
test_dataset = ATCDataSet(testing_df, features, target)
validation_dataset = ATCDataSet(validation_df, features, target)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=20, shuffle=True)


# Build the nn
print(f'input_size: {atc_level5_df[features].iloc[0].shape[0]}')
print(f'output_size: {atc_level5_df[target].iloc[0]}')
input_size, output_size = atc_level5_df[features].iloc[0].shape[0], atc_level5_df[target].iloc[0].shape[0]
neurons_per_node = [80, 80, 80]


class ATCLevel1Classifier(nn.Module):
    def __init__(self, input_size, neurons_per_node: list, output_size, nn_size: int):
        super(ATCLevel1Classifier, self).__init__()
        self.hidden_size = len(neurons_per_node)
        self.nn_size = nn_size

        def check_nn_size():
            if self.hidden_size != self.nn_size:
                raise Exception("the nn_size and len(hiddn_size) do not match. "
                                "Please either edit hidden_size or the nn.Sequential code below to adjust layers")
        check_nn_size()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, neurons_per_node[0]),  # layer 1
            nn.ReLU(),
            nn.Linear(neurons_per_node[0], neurons_per_node[1]),  # layer 2
            nn.ReLU(),
            nn.Linear(neurons_per_node[1], neurons_per_node[2]),  # Layer 3
            nn.ReLU(),
            # nn.Linear(neurons_per_node[2], neurons_per_node[3]),  # Layer 4
            # nn.ReLU(),
            # nn.Linear(neurons_per_node[3], neurons_per_node[4]),  # Layer 5
            # nn.ReLU(),
            # nn.Linear(neurons_per_node[4], neurons_per_node[5]),  # Layer 6
            # nn.ReLU(),
            # nn.Linear(neurons_per_node[5], neurons_per_node[6]),  # Layer 7
            # nn.ReLU(),
            # nn.Linear(neurons_per_node[6], neurons_per_node[7]),  # Layer 8
            # nn.ReLU(),
            nn.Linear(neurons_per_node[2], output_size)
        )

    def forward(self, x):
        x = F.softmax(self.linear_relu_stack(x))  # output layer
        return x


# create an instance of the ATCLevel1Classifier nn
model = ATCLevel1Classifier(input_size, neurons_per_node, output_size, 3).to('cpu')

# adding class weights
class_pcts = training_df['level1'].value_counts(normalize=True)
class_weights = 1 / class_pcts.sort_index()
class_weights = torch.tensor(class_weights.values)

# Hyperparameters
learning_rate, batch_size, epochs = 0.001, 100, 650
loss_fn = nn.BCELoss()  # adding weights seems to help with SGD model!
loss_function = "nn.BCELoss"
optimizer_class = "torch.optim.SGD"
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)


def set_y_hat_tensor_binary(tensor):  # if the softmax value is greater than 0.1 assign 1 else 0
    tensor *= 10
    tensor = torch.floor(tensor)
    tensor /= tensor
    return torch.nan_to_num(tensor)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    average_loss = 0
    number_sampled = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()
        number_sampled += pred.shape[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # get number correct predictions
        bit_vector_y_hat = set_y_hat_tensor_binary(pred)
        equal_tensor = torch.eq(bit_vector_y_hat, y)
        matches = torch.all(equal_tensor, dim=1)
        correct += torch.sum(matches).item()

        if number_sampled == size:
            average_loss = running_loss / (batch + 1)
    return average_loss, (correct, size)


def t_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # get number correct predictions
            bit_vector_y_hat = set_y_hat_tensor_binary(pred)
            equal_tensor = torch.eq(bit_vector_y_hat, y)
            matches = torch.all(equal_tensor, dim=1)
            correct += torch.sum(matches).item()

    test_loss /= num_batches

    return test_loss, (correct, size)


def validation_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()

            # get number correct predictions
            bit_vector_y_hat = set_y_hat_tensor_binary(pred)
            equal_tensor = torch.eq(bit_vector_y_hat, y)
            matches = torch.all(equal_tensor, dim=1)
            correct += torch.sum(matches).item()

    validation_loss /= num_batches

    return validation_loss, (correct, size), y, bit_vector_y_hat


train_loss_per_epoch_dict = {}
train_accuracy_per_epoch_dict = {}
test_loss_per_epoch_dict = {}
test_accuracy_per_epoch_dict = {}
validation_loss_per_epoch = {}
validation_correct_per_epoch = {}
validation_correct_pct_per_epoch = {}
size_batch = 0
for t in tqdm(range(epochs), desc="Epochs:"):
    train_loss, train_correct = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loss_per_epoch_dict[t] = train_loss
    train_accuracy_per_epoch_dict[t] = train_correct[0]/train_correct[1]
    test_loss, test_correct = t_loop(test_dataloader, model, loss_fn)
    test_loss_per_epoch_dict[t] = test_loss
    test_accuracy_per_epoch_dict[t] = test_correct[0]/test_correct[1]
    validation_loss, validation_correct, y, y_bit_vector = validation_loop(validation_dataloader, model, loss_fn)
    validation_loss_per_epoch[t] = validation_loss
    # validation_correct_per_epoch[t] = validation_correct[0]
    validation_correct_pct_per_epoch[t] = validation_correct[0]/validation_correct[1]
    # size_batch = validation_correct[1]

# Saving model performance metrics for plotting
model_assessment_dicts = [train_loss_per_epoch_dict, test_loss_per_epoch_dict, validation_loss_per_epoch,
                          train_accuracy_per_epoch_dict, test_accuracy_per_epoch_dict, validation_correct_pct_per_epoch]
model_assessment_df = pd.DataFrame(model_assessment_dicts)
model_assessment_df.to_csv('multiclass_model.csv')


## if you want to plot from this script

# fig, ax1 = plt.subplots(figsize=(13,7))
#
# ax1.plot(train_loss_per_epoch_dict.keys(), train_loss_per_epoch_dict.values(), label="training_loss", color="red")
# ax1.plot(test_loss_per_epoch_dict.keys(), test_loss_per_epoch_dict.values(), label="test_loss", color="blue")
# ax1.plot(validation_loss_per_epoch.keys(), validation_loss_per_epoch.values(), label="validation_loss", color="green")
# # ax1.plot(validation_correct_pct_per_epch.keys(), validation_correct_pct_per_epch.values(), label="ratio_correct", color="seagreen")
# ax1.set_ylabel("Loss")
# ax2 = ax1.twinx()
# ax2.plot(train_accuracy_per_epoch_dict.keys(), train_accuracy_per_epoch_dict.values(), label="train_accuacy", color="tomato")
# ax2.plot(test_accuracy_per_epoch_dict.keys(), test_accuracy_per_epoch_dict.values(), label="test_accuracy", color="cyan")
# ax2.plot(validation_correct_pct_per_epoch.keys(), validation_correct_pct_per_epoch.values(), label="validation_accuracy", color="lightgreen")
# ax2.set_ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.title(f"learning rate: {learning_rate}, layers: {len(neurons_per_node)},  neurons: {neurons_per_node[0]}, "
#           f"loss_function: {loss_function}, optimizer_class: {optimizer_class} \nepochs: {epochs}, "
#           f" number_per_batch: {size_batch}, classification: {target}, batch_size: {batch_size}")
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# image_directory = '/Users/leoparsons/Desktop/cheminformatics_projects_drafts/ChEMBL rough drafts/Model_Testing_Loss_curves'
# plt.savefig(f"learning rate: {learning_rate}, layers: {len(neurons_per_node)},  neurons: {neurons_per_node[0]}, "
#             f"loss_function: {loss_function}, optimizer_class: {optimizer_class}, epochs: {epochs}, "
#             f" number_per_batch: {size_batch}, classification: {target}, batch_size: {batch_size}.png")
# plt.show()


