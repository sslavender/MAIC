import pickle

with open('datasets/CelebA_train.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)