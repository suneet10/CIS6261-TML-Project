def get_labels():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    label_names = unpickle("batches.meta")

    # print(label_names)

    labels = []
    f = open("label_data_names.txt", "w")
    f.write("fine_label_names\n")
    for label in label_names[b'label_names']:
        labels.append(label)

    return labels