# smarts_pg_query

---

This folder contains two scripts, smarts_pg_query_logic.py which holds some functions and smarts_pg_query_example.py
which shows you a way to use these functions.

---
### Example workflow

smarts_pg_query_example.py walks you through the following steps
1. Connect to the chembl_35 database
2. Query properties of molecules from the database 
3. Get the SMILES strings for those molecules
4. Convert these values to rdkit mol objects
5. Run SMARTS based querries on these objects
6. Visualize the results

___

### Example output

By searching for all molecules in the chembl_35 database which have:

- < 2 hydrogen bonding groups
- and contain both a ketone and benzene ring

We can visualize the results as follows
>![Image of Molecules Matching Query]('/Cheminformatics_projects/ChEMBL to post/SMARTS_pg_query/example.png')

---

### Notes

More practical queries can be made than those shown here. Just a starting point.

---
