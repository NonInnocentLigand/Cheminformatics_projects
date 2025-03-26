from rdkit import Chem
from rdkit.Chem import Draw
import psycopg2


def get_targets(sql: str, psycopg2_config,  data=None, return_as='list', ):
    """
    Use this function to generate a PosgreSQL query from sql and data.
    :param sql: the SQL statement to be parsed by psycopg2.
    :param psycopg2_config: dictionary containing relevant information for psycopg2.connect()
    :param data: variables to bind with the sql statement. Default is None, however if a value is
    assigned to data, it should be a list or tuple. See psycopg2 documentation for more details.
    :param return_as: return the result of the SQL query as a python list or tuple
    :return: result of the SQL query
    """
    target_id_list = []
    with psycopg2.connect(**psycopg2_config) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, data)
            if return_as == 'list':
                row = cur.fetchone()
                target_id_list.append(row[0])
                while row is not None:
                    row = cur.fetchone()
                    try:
                        target_id_list.append(row[0])
                    except TypeError:
                        pass
                return target_id_list
            if return_as == 'tuple':
                row = cur.fetchone()
                target_id_list.append(row[0])
                while row is not None:
                    row = cur.fetchone()
                    try:
                        target_id_list.append(row[0])
                    except TypeError:
                        pass
                return tuple(target_id_list)
    return target_id_list


def parameter_binder_generator(data: list):
    """
    Generate required number of '%s' characters as needed to bind variables to SQL statement
    :param data: python list containing the variables that should be bound to a SQL statement
    :return: python list object
    """
    data_copy = list(data)
    for index in range(len(data)):
        data_copy[index] = '%s'
    return tuple(data_copy)


def match_targets(sql, psycopg2_config, data=None):
    """
    Return list of items from a column of a pg table whose values match those in a separate column of the same table
    :param sql: the SQL statement to be parsed by psycopg2.
    :param psycopg2_config: dictionary containing relevant information for psycopg2.connect() method
    :param data: variables to bind with the sql statement. Default is None, however if a value is
    assigned to data, it should be a list or tuple. See psycopg2 documentation for more details.
    :return: python list object
    """
    return_list = []
    with psycopg2.connect(**psycopg2_config) as conn:
        with conn.cursor() as cur:
            parameters = parameter_binder_generator(data)
            sql = sql + f' {parameters}'
            cur.execute(sql, data)
            row = cur.fetchone()
            return_list.append(row[0])
            while row is not None:
                row = cur.fetchone()
                try:
                    return_list.append(row[0])
                except TypeError:
                    pass
        return return_list


def search_smarts_string(smarts_string: str, list_of_molecules: list):
    """
    Retrieve matches to a chemical SMARTS query from a list of rdkit molecule objects
    :param smarts_string: chemical SMARTS string
    :param list_of_molecules: python list of rdkit molecule objects
    :return: Python list of rdkit molecule objects which contain a match to the SMARTS query
    """
    pattern = Chem.MolFromSmarts(smarts_string)
    pattern_matched = []
    for molecule in list_of_molecules:
        if molecule.HasSubstructMatch(pattern):
            pattern_matched.append(molecule)
    return pattern_matched


def apply_multiple_smarts_filters(starting_mol_list: list, **kwargs):
    """
    Search for molecules containing multiple functional groups in list of rdkit molecule objects
    :param starting_mol_list: python list of rdkit molecule objects to search through
    :param kwargs: a dictionary object can be bound here to specify functional groups (keys) and their SMARTS string
    (values) which the target molecules must contain
    :return: python list object containing the rdkit molecule object matching SMARTS strings (kwargs[kwarg])
    which match molecules in starting_mol_list
    """
    filtered_list = starting_mol_list
    for kwarg in kwargs:
        smarts_string = kwargs[kwarg]
        filtered_list = search_smarts_string(smarts_string, filtered_list)
    return filtered_list


def draw_molecules(number_of_molecules: int, mols_per_row, molecule_list: list, save_image=False):
    """
    create a png of a subset of molecules from a list
    :param number_of_molecules: number of molecules to include from the molecule_list starting from index 0
    :param mols_per_row: number of molecules displayed in a row
    :param molecule_list: python list object of rdkit molecule objects
    :param save_image: save image to path specified
    :return: a png image
    """
    # displays the index +1 for the molecule and the len of molecule list for each molecule displayed
    legends = [f'{molecule_list.index(x) + 1} of {len(molecule_list)}' for x in
               molecule_list[:number_of_molecules]]
    img = Draw.MolsToGridImage(molecule_list[:number_of_molecules], molsPerRow=mols_per_row, legends=legends)
    if save_image:
        image_path = '/SMARTS_pg_query/example.jpeg'
        img.save(image_path)
        print(f'image saved to: {image_path}')
    img.show()
