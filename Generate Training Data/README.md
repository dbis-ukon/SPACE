Path pattern sampler and enumerator
=========

## Build steps

For building our code we have used cmake. We have also used the Boost library v. 1.85.0. 

## Running the Sampler

To run the sampler you use the sampler executable:
  `./sampler -f [EDGES_FILE] -o [OUTPUT_FILE] <options>`
  
Options include:

-r [INTEGER]	Maximum number of random walks to be executed. Sampler finishes if this number is reached.

-s [INTEGER] 	Maximum size of pattern in terms of edges. Sampler generates patterns of size between 2 and the defined number.

-p [edges|all]	The type of pattern that is sampled. A pattern contains either edge labels only, or both edge and node labels.

-n [FILE_NAME]	Input file containing node labels. If -p is set to edges, then this parameter is ignored.

## Running the Enumerator

To run the enumerator you use the enumerator executable:
  `./enumerator -f [EDGES_FILE] -q [PATTERNS_FILE] -o [OUTPUT_FILE] <options>`
  
Options include:

-s [hm|cm|im]    The semantics used for subgraph matching; hm for homomorphism, cm for cyphermorphism and im for isomorphism.

-p [edges|all]   The type of pattern that is given as input. A pattern contains either edge labels only, or both edge and node labels.

-n [FILE_NAME]   Input file containing node labels. If -p is set to edges, then this parameter is ignored.