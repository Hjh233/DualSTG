import time
import sys

import numpy as np
from phe import paillier
import torch


public_key, private_key = paillier.generate_paillier_keypair()

num = int(1.0)
enc = public_key.encrypt(num)
print(sys.getsizeof(enc))
print(sys.getsizeof(num))

def encrypt_time(embedding):

    plain_emb = [emb.item() for emb in embedding]

    begin = time.time()
    encrypted_emb = [public_key.encrypt(x) for x in plain_emb]
    decrypted_emb = [private_key.decrypt(x) for x in encrypted_emb]
    end = time.time()

    # print(decrypted_emb)
    print('total bytes needed is', sys.getsizeof(encrypted_emb))
    print('total time consumed is', end - begin)
    print('*'*100)

    return end - begin


if __name__ == "__main__":
    # exit(0)

    non_zero_embedding = torch.randn(100)
    zero_embedding = torch.zeros(24, dtype=int)

    mixed_embedding = torch.concat((non_zero_embedding, zero_embedding), dim=0)
    # print(mixed_embedding)

    # mixed_cost = encrypt_time(embedding=mixed_embedding)
    non_zero_cost = encrypt_time(embedding=non_zero_embedding)
    # zero_cost = encrypt_time(embedding=zero_embedding)

    non_zero_cost_array = np.zeros(5)

    for i in range(5):
        non_zero_cost_array[i] = encrypt_time(embedding=non_zero_embedding)

    print(np.sum(non_zero_cost_array) / 5) # Enc/Dec 100-dim tensor takes 3.6402761459350588 seconds

    '''
       emb_size   |   total bytes needed ([enc])
          12      |          184
          24      |          256
          48      |          520
          100     |          904
    '''



    
