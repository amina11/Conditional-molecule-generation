from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

import gzip
import pickle
import sys

import numpy as np

RADIUS = 2
SIZE = 2048

def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bitstring = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, SIZE).ToBitString()
    return np.array([i=='1' for i in bitstring], dtype=bool)

def precompute_fingerprints(infile, outfile):
    output = {}
    for row in open(infile):
        smiles = row.split(',')[1]
        bitstring = fingerprint(smiles)
        output[smiles] = bitstring
    pickle.dump(output, gzip.open(outfile, 'wb'))
    return output

def load_training_logP():
    all_logP = []
    all_smiles = []
    for row in open("data/qm9.csv"):
        row = row.strip().split(',')
        all_smiles.append(row[1])
        all_logP.append(float(row[4]))
    return np.array(all_logP), np.array(all_smiles)
        
def similarity(fp1, fp2):
    return np.logical_and(fp1, fp2).sum() / np.logical_or(fp1, fp2).sum()

def smiles_similarity(smiles1, smiles2):
    return similarity(fingerprint(smiles1), fingerprint(smiles2))

class Baseline(object):

    def __init__(self, data_source="data/qm9.csv"):
        self.all_logP, self.all_smiles = load_training_logP()
        outfile = "%s.pkl.gz" % data_source
        try:
            self.dataset_fingerprints = pickle.load(gzip.open(outfile, "rb"))
        except:
            print("pre-computing fingerprints for training dataset")
            self.dataset_fingerprints = precompute_fingerprints(data_source, outfile)


    def find_closest(self, smiles, target_logP, topK=5, tolerance=0.05):
        target_fingerprint = fingerprint(smiles)
        if smiles in self.dataset_fingerprints:
            assert (target_fingerprint == self.dataset_fingerprints[smiles]).all()
        filtered = np.abs(self.all_logP - target_logP) < np.abs(target_logP*tolerance)
        self.filtered = filtered
        candidates = [self.dataset_fingerprints[s] for s in self.all_smiles[filtered]]
        score = np.array([similarity(target_fingerprint, c) for c in candidates])
        best = score.argsort()[-topK:]
        return self.all_smiles[filtered][best[::-1]]


if __name__ == '__main__':
    target_smiles = sys.argv[1]
    target_logP = float(sys.argv[2])
    mol = Chem.MolFromSmiles(target_smiles)
    logP = Descriptors.MolLogP(mol)

    print("Looking for molecules similar to %s (logP = %0.3f), but with logP = %0.3f" % (target_smiles, logP, target_logP))
#     Draw.MolToImage(mol).save(target_smiles+".png")

    baseline = Baseline()
    best = baseline.find_closest(target_smiles, target_logP)
    
    mols = [mol]
    
    print("\nTop 5 found molecules in dataset:\n")
    
    for smiles in best:
        mol = Chem.MolFromSmiles(smiles)
        logP = Descriptors.MolLogP(mol)
        print("%s (logP = %0.3f, similarity = %0.2f)" % (smiles, logP, smiles_similarity(target_smiles, smiles)))
        mols.append(mol)

    print("\nSaving image of original molecule plus 5 candidates to %s.png" % target_smiles)
    Draw.MolsToImage(mols).save(target_smiles+".png")
