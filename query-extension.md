INPUT: S = [source codes], T = [target codes], SN = SNOMED-CT
OUTPUT: M = {maps from s -> T}

Initialize: M = []

For all s in S:
    A_code = s['code']
    A_desc = s['desc']
    A = preprocess(A_desc)

    possible_maps = []

    For all t in T:
        B_code = t['code']
        B_desc = t['desc']
        B = preprocess(B_desc)

        A_not_B = A - B
        B_not_A = B - A

        if len(A_not_B) == 0 or len(B_not_A) == 0:
            possible_maps.clear()
            possible_maps.append({
                B_code,
                B_desc
            })
            break
        
        A_not_B_concepts = SN.search(A_not_B)
        B_not_A_concepts = SN.search(B_not_A)

        common_concepts = A_not_B_concepts & B_not_A_concepts

        if len(common_concepts) == 0:
            continue
        
        possible_maps.apend({
            B_code,
            B_desc,
            common_concepts
        })
    
    M.apend({
        src: {
            A_code,
            A_desc
        },
        possible_maps
    })

return M


















INPUT: list[code description, __hierarchical_info__]
OUTPUT: list[equivalent/similar terms to given code description]



Initialize: Bucket[]

step1: preprocessing
    1. Bucket.append(code descriptions), Bucket.append(__hierarchical_info__)
    2. normalize and remove commas from the code description.
    3. tokens = word_tokenize(code descriptions)
    4. remove any stop words including other, due, specified, unspecified and so on.

step2: query extension
    1. select all the nouns (proper nouns) from the tokens.
    2. create a query string with all possible combinations = queries
    3. Bucket.append(queries)

step3: For each query in queries:
        terms = get similar terms using SNOMED-CT
        Bucket.append(terms)

step4: return Bucket













Foundation models, for eg. in language like BERT, GPT, that are trained on large training corpus in a self-supervised fashion, can be used in any downstream (language) tasks with/without fine-tuning. This is possible because they share common vocabulary. But with graphical data e.g. Knowledge Graphs, have non overlaping sets of nodes and edges. This poses a key challenge in building a foundation model for graph reasoning.

Inferencing Methods:
    1. Transductive Inference (same sets of nodes and relations/edges during inferencing compared to training);
    2. Inductive Inference (Different sets of nodes or relations, or both during inferencing compared to training);

Most Inductive Inference methods assumes a fixed set of either nodes or relations. Different to this, ULTRA (unified, learnable, and transferable) aim to do graph reasoning with arbitary set of nodes and relations. The key ingredient here is an invariable relational structures (although the relations themselves could be different but the interaction between them could be similar and transferable).