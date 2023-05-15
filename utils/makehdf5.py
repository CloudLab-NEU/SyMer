import h5py
import tqdm

if __name__ == '__main__':
    index = 0
    role = "test"
    with h5py.File(f"../data/deepcom.{role}.h5", 'w', libver='latest') as h5f:
        with open(f"../data/deepcom.{role}", 'r') as cf:
            for line in tqdm.tqdm(cf):
                h5f.create_group(str(index))
                h5f[str(index)].create_dataset('line', data=line)
                index += 1
