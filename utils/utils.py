from sklearn.preprocessing import LabelEncoder

def get_labels(items):
    le = LabelEncoder()
    le.fit(items)

    return (le.transform, le.inverse_transform)


def logger(path, text):
    """
    Function to write text logs to the given file

    Args:
        path (string): log file location
        text (string): log text to write to log file
    """

    with open(path, 'a') as f:
        f.write(text)
