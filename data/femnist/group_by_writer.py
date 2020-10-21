import os
import pickle


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

wwcd = os.path.join('intermediate', 'write_with_class')
write_class = load_obj(wwcd)

writers = []  # each entry is a (writer, [list of (file, class)]) tuple
cimages = []
(cw, _, _) = write_class[0]
for (w, f, c) in write_class:
    if w != cw:
        writers.append((cw, cimages))
        cw = w
        cimages = [(f, c)]
    cimages.append((f, c))
writers.append((cw, cimages))

ibwd = os.path.join('intermediate', 'images_by_writer')
save_obj(writers, ibwd)