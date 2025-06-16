## Managing Data Uncertainty in Automatic Mapping of Clinical Classification Systems

## Abstract
Mapping clinical classification systems, like the International Classification of Disease (ICD) across different versions and other external clinical classifications systems, is challenging and often done manually by trained professionals. Among others, variation in the code descriptions to describe the same clinical condition in different versions poses a unique challenge to implementing automated mapping systems. We call this *data uncertainty*. Existing lexical-based methods attempt to solve this problem by generating alternative terms using synonyms. This work addresses the data uncertainty by learning a *probabilistic embedding* for each code description using similar terms and paraphrases. A valid code pair must exhibit proximity in the embedding space and have a comparable distribution. Additionally, we propose a new evaluation metric that considers the hierarchical structure of ICD to evaluate the performance of an automated mapping system. We demonstrate the effectiveness of our approach by mapping ICD-9-CM (Clinical Modification) and ICD-10-CM, ICD-10-AM (Australian Modification) and ICD-11 in both directions.

[Paper Link](https://rdcu.be/eq62S)

## Usage
To compute the *Hierarchical Similarity Score* use the `compute_scores.py` script.
```console
python compute_scores.py --data-path <path_to_csv_data_file> --out-path <output_file.pkl>
```

To generate ICD code description embedding use the `gen_embeddings.py` script. Example, you can use the following command to generate embedding using the term only.
```console
python gen_embeddings.py --input <path_to_csv_data_file>
```
If you want to generate embedding using the similar terms, then add `--use-similar-terms` and set `--d` option to specify the number of similar terms to use.

To generate the maps, use the `main.py` script.

**Important: Please make sure that the ground-truth files are in the folder `./data/gt` and the filename should be `<source>_<tgt>.pkl`. For example, for the mapping from icd9cm to icd10cm, the ground-truth file should be `./data/gt/icd9cm_icd10cm.pkl`.**
```console
python main.py --src "./data/icd9cm.csv" --tgt "./data/icd10cm.csv" --d 30 --compute-statistics
```

## Citation
```
@InProceedings{10.1007/978-981-96-8298-0_23,
author="Purja Pun, Santosh
and Obst, Oliver
and Basilakis, Jim
and Ginige, Jeewani Anupama",
editor="Wu, Xintao
and Spiliopoulou, Myra
and Wang, Can
and Kumar, Vipin
and Cao, Longbing
and Zhou, Xiangmin
and Pang, Guansong
and Gama, Joao",
title="Managing Data Uncertainty in Automatic Mapping of Clinical Classification Systems",
booktitle="Data Science: Foundations and Applications",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="284--295",
abstract="Mapping clinical classification systems, like the International Classification of Disease (ICD) across different versions and other external clinical classifications systems, is challenging and often done manually by trained professionals. Among others, variation in the code descriptions to describe the same clinical condition in different versions poses a unique challenge to implementing automated mapping systems. We call this data uncertainty. Existing lexical-based methods attempt to solve this problem by generating alternative terms using synonyms. This work addresses the data uncertainty by learning a probabilistic embedding for each code description using similar terms and paraphrases. A valid code pair must exhibit proximity in the embedding space and have a comparable distribution. Additionally, we propose a new evaluation metric that considers the hierarchical structure of ICD to evaluate the performance of an automated mapping system. We demonstrate the effectiveness of our approach by mapping ICD-9-CM (Clinical Modification) and ICD-10-CM, ICD-10-AM (Australian Modification) and ICD-11 in both directions. The source code will be available at: https://github.com/Xujan24/wt-KL",
isbn="978-981-96-8298-0"
}
```

