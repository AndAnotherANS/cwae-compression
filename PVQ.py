import itertools
import json
import math
import os.path

import numpy as np

def round_to_3(x):
    n_digits = -int(math.floor(math.log10(abs(x)))) + 2
    return round(x, n_digits)

class PVQ:
    def __init__(self, n, k, cache_dir=None):
        self.n = n
        self.k = k
        if cache_dir is not None:
            cache_dir2 = os.path.join(cache_dir, f"{self.n}_{self.k}.json")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            if os.path.exists(cache_dir2):
                    self._load_cached_codebook(cache_dir2)
            else:
                self._generate_codebook()
                self._cache_codebook(cache_dir2)
        else:
            self._generate_codebook()

        self.least_power_2 = 1
        while 2 ** self.least_power_2 < len(self.codebook):
            self.least_power_2 += 1

    def _generate_codebook(self):
        self.codebook = dict()
        i = 0
        for p in itertools.product(range(-self.k, self.k + 1), repeat=self.n):
            if sum(abs(x) for x in p) == self.k:
                self.codebook[i] = np.array(p) / np.linalg.norm(np.array(p), 2, 0)
                i += 1

    def _get_closest_code(self, v):
        min_dist = 1e20
        best_code = -1
        for code, p in self.codebook.items():
            dist = np.sqrt(np.square(v - p).sum(0))
            if dist < min_dist:
                min_dist = dist
                best_code = code
        return best_code

    def encode(self, vector):
        codes = np.zeros([math.floor(vector.shape[0]/(self.n + 1e-10)) + 1]).astype(int)
        norms = [0] * (math.floor(vector.shape[0]/(self.n + 1e-10)) + 1)
        for i in range(codes.shape[0]):
            vector_slice = vector[i*self.n:(i+1)*self.n]
            norm = np.linalg.norm(vector_slice, 2, -1)
            normalized = vector_slice/(norm + 1e-20)
            code = self._get_closest_code(normalized)
            codes[i] = code
            norms[i] = bin(np.float16(norm).view('H'))[2:].zfill(16)

        entry = ""
        for code, norm in zip(codes, norms):
            code_binary = np.binary_repr(code)
            while len(code_binary) < self.least_power_2:
                code_binary = "0" + code_binary

            entry += code_binary + norm

        return entry

    def decode(self, entry):
        vector = []
        while entry:
            code, norm, entry = self._get_codes_norms(entry)
            vector.append(self.codebook[code] * norm)
        vector = np.expand_dims(np.concatenate(vector), 0)
        return vector

    def _cache_codebook(self, cache_dir):
        codebook_lists = {k: v.tolist() for k, v in self.codebook.items()}
        with open(cache_dir, "w+") as file:
            json.dump(codebook_lists, file)

    def _load_cached_codebook(self, cache_dir):
        with open(cache_dir, "r") as file:
            contents = json.load(file)
        codebook_numpy = {int(k): np.array(v) for k, v in contents.items()}
        self.codebook = codebook_numpy

    def _get_codes_norms(self, entry):
        code, norm = entry[:self.least_power_2], entry[self.least_power_2:self.least_power_2 + 16]
        code = int(code, 2)
        norm = np.array((int(norm, 2)), dtype="H").view(np.float16)
        entry = entry[self.least_power_2 + 16:]
        return code, norm, entry


if __name__ == '__main__':
    vectors = np.random.normal(0, 1, [3000])
    for k in range(10, 20):
        pvq = PVQ(3, k, cache_dir="./data/pvq")
        encoded = pvq.encode(vectors)
        decoded = pvq.decode(encoded)
        print(np.linalg.norm(vectors - decoded).sum(-1).mean())