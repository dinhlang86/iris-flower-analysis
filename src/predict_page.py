import pickle


def load_model():
    with open('svm_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


# Extract svm model, label encoder, standard scaler
data = load_model()
svm_model = data['model']
le = data['le']
scaler = data['scaler']
