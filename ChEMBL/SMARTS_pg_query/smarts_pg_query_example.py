from rdkit import Chem
from rdkit.Chem import Draw
import psycopg2
from smarts_pg_query_functions import get_targets, parameter_binder_generator, match_targets, search_smarts_string, \
    apply_multiple_smarts_filters, draw_molecules

# dictionary containing relevant information to connect to a postgresql database using psycopg2.connect() method
# Fill in your own values as needed
config_dic = {'dbname': '',
              'user': '',
              'password': '',
              'host': '',
              'port': 0000}

# targets = list of molregno values for molecules with less than two hydrogen bonds
targets = get_targets(sql='SELECT DISTINCT molregno FROM compound_properties WHERE hba_lipinski < (%s)',
                      psycopg2_config=config_dic, data=(2, ), return_as='tuple')

target_smiles = match_targets(sql='SELECT canonical_smiles FROM compound_structures WHERE molregno IN',  # make sure
                              # there is no trailing space after the last sql clause
                              psycopg2_config=config_dic, data=targets)

# generate a list of rdkit molecule objects from the list of smiles strings
target_mols = list(map(lambda x: Chem.MolFromSmiles(x), target_smiles))


# Dictionary used to filter results based on functional group
smiles_substructure_dic = {
    'ketone': '[#6][CX3](=O)[#6]',
    'aromatic_ring': 'c1ccccc1',
}

pattern_matches = apply_multiple_smarts_filters(starting_mol_list=target_mols, **smiles_substructure_dic)

draw_molecules(16, 4, pattern_matches, save_image=True)

