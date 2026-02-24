# Developed by Namphil Kim
# Last updated: 2026-02-24

import sys
import os
import time
import BCRmap

help_text = 'How to use BCRmap\n\t-i/--input_file: BCR repertoire file in tsv format, see reference_files/test_file.tsv\n'
help_text += '\t-o/--output_dir: The directory to save results. File name is written automatically according to the input file\n'
help_text += '\t--label_name: The name of additional labels, seperated by "+" for multiple cases (optional)\n'
help_text += '\t--label_file: The path to the label files, seperated by "+" for multiple cases (optional), see reference_files/CoV_AbDab_clonotypes.csv\n'
help_text += '\t--absolute_vj: Set "True" for fixed VJ gene positions (Default=False)\n'
help_text += '\t--vj_reference: The file for the VJ gene usage reference, see reference_files/vaccination_repertoire_VJ_combination_summarized.csv (Needed if --absolute_vj True)\n'
help_text += '\t--save_html: Option to save results in interactive HTML format (Default=True)\n'
help_text += '\t--save_png: Option to save results in non-interactive PNG format (Default=False)\n'
help_text += '\t--dpi: DPI value when saving in PNG (Default=100)\n'
help_text += '\t--show_id: Option to display meta information (Default=True)\n'
help_text += '\t--naive: Retain naive clonotypes (Default=False)\n'

def run_BCRmap():
    input_categories = [['-h', '--help'],
                        ['-i', '--input_file'],
                        ['-o', '--output_dir'],
                        ['--label_name'],
                        ['--label_file'],
                        ['--absolute_vj'],
                        ['--vj_reference'],
                        ['--save_html'],
                        ['--save_png'],
                        ['--dpi'],
                        ['--show_id'],
                        ['--naive']]

    input_list = sys.argv[1:]
    option_dict = {'-i': '', '-o': '', '--label_name':'', '--label_file':'','--absolute_vj':False, '--vj_reference': '',
                   '--save_html':True, '--save_png':False,'--dpi':500, '--show_id':True,'--naive':False}
    for i, input in enumerate(input_list):
        for h, header in enumerate(input_categories):
            if input in header and h == 0:
                print(help_text)
                return
            elif input in header and h != 0:
                option_dict[header[0]] = input_list[i + 1]
                if input_list[i + 1] == 'All' and h == 4:
                    option_dict[header[0]] = 'IGHM_IGHD_IGHG1_IGHG2_IGHG3_IGHG4_IGHGP_IGHA1_IGHA2_IGHE'

    input_file = option_dict['-i']
    save_dir = option_dict['-o']
    start_time = time.time()

    label_file_list = option_dict['--label_file'].split('+')
    label_name_list = option_dict['--label_name'].split('+')

    bool_dict = {'True': True, 'False': False, True: True, False: False}
    file = input_file.split('/')[-1]
    name = file.replace('.tsv', '').replace('_e1_a1_f1_c1', '').replace('_merged', '').replace('_add_d_gene', '')
    clonotype_dict, v_dict, clonality, num_clonotype, total_reads, total_isotypes = BCRmap.calibrate_clonotype(input_file,bool_dict[option_dict['--naive']])
    print("--- %s seconds passed during preprocessing ---" % (time.time() - start_time))
    metatext = 'Total reads: ' + str(total_reads) + '<br>Total clonotypes: ' + str(
        num_clonotype) + '<br>Clonality: ' + "%.3f" % clonality + '<br>IgM/D: ' + "%.1f" % (
                           100 * (total_isotypes[0]+total_isotypes[1])) + '%  IgG: ' + "%.1f" % (
                           100 * (total_isotypes[2]+total_isotypes[3]+total_isotypes[5]+total_isotypes[6])) + '%  IgA: ' + "%.1f" % (
                           100 * (total_isotypes[4]+total_isotypes[7]+total_isotypes[8])) + '%'

    if bool_dict[option_dict['--absolute_vj']]:
        vj_angle_dict = BCRmap.get_vj_angle_absolute(option_dict['--vj_reference'])
        name += '_abs'
    else:
        vj_angle_dict = BCRmap.get_vj_angle(v_dict)

    BCRmap.new_node_and_edge(label_file_list, label_name_list, clonotype_dict, vj_angle_dict, name, save_dir+name, metatext,total_isotypes,
                  save_html=bool_dict[option_dict['--save_html']], save_png=bool_dict[option_dict['--save_png']], dpi=option_dict['--dpi'], show_id=bool_dict[option_dict['--show_id']])
    print("--- %s seconds passed during image generation ---" % (time.time() - start_time))
    return

if __name__ == '__main__':
    run_BCRmap()