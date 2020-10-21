import os
import pickle


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


cfhd = os.path.join('intermediate', 'class_file_hashes')
wfhd = os.path.join('intermediate', 'write_file_hashes')
class_file_hashes = load_obj(cfhd) # each elem is (class, file dir, hash)
write_file_hashes = load_obj(wfhd) # each elem is (writer, file dir, hash)

class_hash_dict = {}
for i in range(len(class_file_hashes)):
    (c, f, h) = class_file_hashes[len(class_file_hashes)-i-1]
    class_hash_dict[h] = (c, f)

write_classes = []
for tup in write_file_hashes:
    (w, f, h) = tup
    write_classes.append((w, f, class_hash_dict[h][0]))

wwcd = os.path.join('intermediate', 'write_with_class')
save_obj(write_classes, wwcd)