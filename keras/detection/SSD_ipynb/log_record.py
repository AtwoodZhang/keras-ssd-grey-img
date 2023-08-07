import pickle


# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
def record_log(history, filename="log.txt"):
    with open(filename, "wb") as f:
        pickle.dump(history.history, f)
        

def read_log(filename="log.txt"):
    with open(filename, 'rb') as f:
        history = pickle.load(f)
        return history

