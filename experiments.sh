#!/bin/bash

python experiments.py --match any --distance_function hamming --hash_method trivial,lsh,none --hash_length 32,768
