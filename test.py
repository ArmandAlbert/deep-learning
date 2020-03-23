import pickle

x = {'a': 1, 'b': 2}

with open('1.pickle', 'rb') as op:
    data = pickle.load(op)
    print(data)
    op.close
