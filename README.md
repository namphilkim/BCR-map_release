# BCR-map_release
Tool for visualizing the BCR repertoire into an interactive image

Dependencies of BCRmap

mpld3,matplotlib,numpy,BCR-SORT

How to run BCRmap

Example code: 

python run.py -i reference_files/test_file.tsv -o result/ --label_name CovAbDab+Influenza --label_file reference_files/CoV_AbDab_clonotypes.csv+/reference_files/Influenza_clonotypes.csv --show_id True --save_html True --save_png True --absolute_vj True --vj_reference reference_files/vaccination_repertoire_VJ_combination_summarized.csv --naive False

Current issues: Interactive legend does not run properly without giving extra labels, so adding CovAbDab as a default is advised 
