import hashlib
import os
import pickle


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


cfd = os.path.join('intermediate', 'class_file_dirs')
wfd = os.path.join('intermediate', 'write_file_dirs')

class_file_dirs = load_obj(cfd)
write_file_dirs = load_obj(wfd)

class_file_hashes = []
write_file_hashes = []

count = 0
for tup in class_file_dirs:
    if count % 100000 == 0:
        print('hashed %d class images' % count)

    (cclass, cfile) = tup
    file_path = os.path.join(cfile)

    chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    class_file_hashes.append((cclass, cfile, chash))

    count += 1

cfhd = os.path.join('intermediate', 'class_file_hashes')
save_obj(class_file_hashes, cfhd)

count = 0
for tup in write_file_dirs:
    if (count % 100000 == 0):
        print('hashed %d write images' % count)

    (cclass, cfile) = tup
    file_path = os.path.join(cfile)

    chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    write_file_hashes.append((cclass, cfile, chash))

    count += 1

wfhd = os.path.join('intermediate', 'write_file_hashes')
save_obj(write_file_hashes, wfhd)