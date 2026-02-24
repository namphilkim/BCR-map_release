import os
import random as rd
import numpy as np
import torch
from torch.utils.data import DataLoader
import mpld3
from mpld3 import plugins,utils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from functools import partial
import chord_diagram as chord
from model_utils import predict
from data_utils import sequence_padding, generate_dict, decode_sequence, decode_categorical
from data_loader import DatasetBCRSORT
from model import load_model

class PointClickInfo(plugins.PluginBase):
    """Plugin: clicking a point updates info in a HTML div below the plot."""
    JAVASCRIPT = """
        mpld3.register_plugin("pointclickinfo", PointClickInfo);
        PointClickInfo.prototype = Object.create(mpld3.Plugin.prototype);
        PointClickInfo.prototype.constructor = PointClickInfo;
        PointClickInfo.prototype.requiredProps = ["idpts", "labels", "divid",
                                                    "edge_color", "highlight_edge_color",
                                                      "edge_width", "highlight_edge_width"];

        function PointClickInfo(fig, props){
            mpld3.Plugin.call(this, fig, props);
        };

        PointClickInfo.prototype.draw = function(){
            var pts = mpld3.get_element(this.props.idpts, this.fig);
            var labels = this.props.labels;
            var divid = this.props.divid;
            var ec = this.props.edge_color;
            var hec = this.props.highlight_edge_color;
            var ew = this.props.edge_width;
            var hew = this.props.highlight_edge_width;

            // Create container div if not exists
            var parent = d3.select("body").select("#mpld3-side-wrapper");
            if (parent.empty()) {
                parent = d3.select("body")
                           .append("div")
                           .attr("id", "mpld3-side-wrapper")
                           .style("display", "flex")
                           .style("flex-direction", "row")
                           .style("align-items", "flex-start");
                // Move the fig's container into this wrapper
                var figdiv = d3.select("body").select("div[id^='fig_']:first-child");
                parent.node().appendChild(figdiv.node());
                // Append the info panel
                parent.append("div")
                      .attr("id", divid)
                      .style("flex", "0 0 12%")
                      .style("margin-left", "1px")
                      .style("margin-top", "500px")
                      .style("padding", "5px")
                      .style("border", "1px solid #ccc")
                      .style("background-color", "#f9f9f9");
            }
            var container = d3.select("#" + divid);

            // On click: update the div with label for clicked point
            var last_index = null;
            pts.elements().on("mousedown", function(d, i){
                if (last_index !== null) {
                    pts.elements()
                       .filter(function(dd, ii){ return ii === last_index; })
                       .style("stroke", ec)
                       .style("stroke-width", ew);
                }
                container.html(labels[i]);
                d3.select(this)
                  .style("stroke", hec)
                  .style("stroke-width", hew);
                last_index = i;
                
            });
        };
        """

    def __init__(self, points, labels, divid="point-info",
                 edge_color="black", highlight_edge_color="red",
                 edge_width=1, highlight_edge_width=4,
                 css=None):
        self.dict_ = {
            "type": "pointclickinfo",
            "idpts": utils.get_id(points),
            "labels": labels,
            "divid": divid,
            "edge_color": edge_color,
            "highlight_edge_color": highlight_edge_color,
            "edge_width": edge_width,
            "highlight_edge_width": highlight_edge_width
        }
        if css is not None:
            self.css_ = css

def hamming_dist(seq1, seq2):
    dist = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist += 1
    return dist

def kidera_dist(seq1,seq2):
    kidera_dict = {'A':np.array([-1.56,-1.67,-0.97,-0.27,-0.93,-0.78,-0.2,-0.08,0.21,-0.48]),
                   'R':np.array([0.22,1.27,1.37,1.87,-1.7,0.46,0.92,-0.39,0.23,0.93]),
                   'N':np.array([1.14,-0.07,-0.12,0.81,0.18,0.37,-0.09,1.23,1.1,-1.73]),
                   'D':np.array([0.58,-0.22,-1.58,0.81,-0.92,0.15,-1.52,0.47,0.76,0.7]),
                   'C':np.array([0.12,-0.89,0.45,-1.05,-0.71,2.41,1.52,-0.69,1.13,1.1]),
                   'Q':np.array([-0.47,0.24,0.07,1.1,1.1,0.59,0.84,-0.71,-0.03,-2.33]),
                   'E':np.array([-1.45,0.19,-1.61,1.17,-1.31,0.4,0.04,0.38,-0.35,-0.12]),
                   'G':np.array([1.46,-1.96,-0.23,-0.16,0.1,-0.11,1.32,2.36,-1.66,0.46]),
                   'H':np.array([-0.41,0.52,-0.28,0.28,1.61,1.01,-1.85,0.47,1.13,1.63]),
                   'I':np.array([-0.73,-0.16,1.79,-0.77,-0.54,0.03,-0.83,0.51,0.66,-1.78]),
                   'L':np.array([-1.04,0,-0.24,-1.1,-0.55,-2.05,0.96,-0.76,0.45,0.93]),
                   'K':np.array([-0.34,0.82,-0.23,1.7,1.54,-1.62,1.15,-0.08,-0.48,0.6]),
                   'M':np.array([-1.4,0.18,-0.42,-0.73,2,1.52,0.26,0.11,-1.27,0.27]),
                   'F':np.array([-0.21,0.98,-0.36,-1.43,0.22,-0.81,0.67,1.1,1.71,-0.44]),
                   'P':np.array([2.06,-0.33,-1.15,-0.75,0.88,-0.45,0.3,-2.3,0.74,-0.28]),
                   'S':np.array([0.81,-1.08,0.16,0.42,-0.21,-0.43,-1.89,-1.15,-0.97,-0.23]),
                   'T':np.array([0.26,-0.7,1.21,0.63,-0.1,0.21,0.24,-1.15,-0.56,0.19]),
                   'W':np.array([0.3,2.1,-0.72,-1.57,-1.16,0.57,-0.48,-0.4,-2.3,-0.6]),
                   'Y':np.array([1.38,1.48,0.8,-0.56,0,-0.68,-0.31,1.03,-0.05,0.53]),
                   'V':np.array([-0.74,-0.71,2.04,-0.4,0.5,-0.81,-1.07,0.06,-0.46,0.65])}
    dist = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist += np.sum(kidera_dict[seq1[i]]-kidera_dict[seq2[i]]) #not squared or rooted to reduce calculation time and have positive and negative range
    return dist

def calibrate_clonotype(file,naive):  # Extract repertoire charactertics for drawing BCR map
    isotype_dict = {'IGHM': 0, 'IGHD': 1, 'IGHG3': 2, 'IGHG1': 3, 'IGHA1': 4, 'IGHG2': 5, 'IGHG4': 6, 'IGHGP': 3, 'IGHE': 7, 'IGHA2': 8,
                    'M': 0, 'D': 1, 'G3': 2, 'G1': 3, 'A1': 4, 'G2': 5, 'G4': 6, 'GP': 3, 'E': 7, 'A2': 8,
                    'G':5,'A':8}
    clonotype_dict = {}  # Information dictionary on each clonotype
    v_dict = {}  # Frequency info on VJ gene usage
    file_pass = open(file, 'r')
    no_freq = False
    total_reads = 0  # Normalizing factor when frequency column does not exist
    throughput = 0  # Total sequencing reads for each repertoire
    total_isotypes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Cumulative frequency distribution for each isotype group + naive
    for l, line in enumerate(file_pass):  # Reading processed BCR file
        split_line = line.split('\t')
        list_line = split_line[:-1] + [split_line[-1][:-1]]
        if l > 0:
            v = list_line[v_index].split('*')[0]
            j = list_line[j_index].split('*')[0]
            cdr3 = list_line[cdr3_index]
            freq = float(list_line[freq_index])
            total_reads += freq
            throughput += int(list_line[read_index])
            isotype = list_line[c_index].split('*')[0]
            shm = int(list_line[shm_index])
            if isotype in ['IGHM', 'IGHD','M','D'] and shm < 2:
                if naive == False:
                    continue
                else:
                    total_isotypes[-1] += freq
            clonotype = v + '_' + j + '_' + cdr3
            if v not in v_dict and v != None:
                v_dict[v] = [0, 0, 0, 0, 0, 0]
            v_dict[v][int(j[-1]) - 1] += freq
            if clonotype not in clonotype_dict:
                clonotype_dict[clonotype] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # No. seq, freq, M/D, G, A, SHM, random x/y
            clonotype_dict[clonotype][0] += 1
            clonotype_dict[clonotype][1] += freq
            clonotype_dict[clonotype][2 + isotype_dict[isotype]] += freq
            total_isotypes[isotype_dict[isotype]] += freq
            clonotype_dict[clonotype][11] += shm
            seed = int.from_bytes(clonotype.encode('utf-8'), byteorder='big', signed=False)
            rd.seed(seed)
            clonotype_dict[clonotype][12] = rd.random()
            clonotype_dict[clonotype][13] = rd.random()  # Random XY for future coordinate positioning. 0~1

        else:
            v_index = list_line.index('v_call')
            j_index = list_line.index('j_call')
            c_index = list_line.index('c_call')
            cdr3_index = list_line.index('cdr3_aa')
            shm_index = list_line.index('v_alignment_mutation')
            try:
                freq_index = list_line.index('frequency')  # try if the file has a "frequency column"
            except ValueError:
                freq_index = list_line.index('duplicate_count')
                no_freq = True
            read_index = list_line.index('duplicate_count')

    num_clonotype = len(clonotype_dict)
    print('Clonotype calibration finished')
    print('Total number of clonotypes: ' + str(num_clonotype))

    freq_list = []

    for key in clonotype_dict:
        if no_freq:  # Normalize to frequency if frequency column does not exist
            for i in range(1,11):
                clonotype_dict[key][i] = clonotype_dict[key][i] / total_reads  # Freuqency

        freq_list.append(clonotype_dict[key][1])  # Needed to calculate clonality of repertoire

    if no_freq:  # Calculating cumulative frequency of each isotype group
        for i in range(9):
            total_isotypes[i] = total_isotypes[i] / total_reads

        for key in v_dict:  # Calculating distribution for each VJ gene
            for i in range(6):
                v_dict[key][i] = v_dict[key][i] / total_reads

    sorted_freq_list = sorted(freq_list)
    bottom = 0
    clonality = 0
    for f, freq in enumerate(sorted_freq_list):  # Calculating GINI index(clonality)
        clonality += 2 * ((f + 1) / num_clonotype - (bottom + freq)) / num_clonotype
        bottom += freq

    return clonotype_dict, v_dict, clonality, num_clonotype, throughput, total_isotypes


def read_csv(file_name):
    file_pass = open(file_name, 'r')
    list_line_pass = []
    split_line_pass = []
    for l, line in enumerate(file_pass):
        split_line_pass = line.split(',')
        list_line_pass.append(split_line_pass[:-1] + [split_line_pass[-1][:-1]])
    return list_line_pass

def get_vj_angle_absolute(reference_file):
    vj_reference_file = reference_file
    temp_list = read_csv(vj_reference_file)
    vj_angle_dict = {}

    vj_list = []
    for l, line in enumerate(temp_list):
        if l == 0:
            for i in range(1, len(line)):
                vj_angle_dict[line[i]] = []
            vj_list += line[1:]
        else:
            for i in range(1, len(line)):
                vj_angle_dict[vj_list[i - 1]].append(float(line[i]))

    sorted_angle_dict = {}
    for key in vj_angle_dict:
        sorted_angle_dict[key] = sum(vj_angle_dict[key]) / len(vj_angle_dict[key])

    sorted_angle_dict = dict(sorted(sorted_angle_dict.items(), key=lambda x: (x[0])))

    start = 0.0
    vj_angle_dict = {}
    for key in sorted_angle_dict:
        vj_angle_dict[key] = [start, start + 2 * np.pi * sorted_angle_dict[key]]
        start += 2 * np.pi * sorted_angle_dict[key]
    return vj_angle_dict

def get_vj_angle(v_dict):  # Setting angles for the outer VJ gene chords
    ordered_list = []
    total = 0
    for i in range(1, 8):  # V gene families from IGHV1 to IGHV7
        ordered_list.append([])
        str_v_gene = []
        for key in v_dict:
            if key[4] == str(i):  # Create numerically sorted V gene list for each family
                total += sum(v_dict[key])
                if ordered_list[-1] == []:
                    ordered_list[-1].append(key)
                else:
                    index = len(ordered_list[-1])
                    found = False
                    for v, v_gene in enumerate(ordered_list[-1]):
                        try:
                            if int(v_gene.split('-')[1][:2]) < int(key.split('-')[1][:2]) and found == False:
                                continue
                            elif int(v_gene.split('-')[1][:2]) > int(key.split('-')[1][:2]) and found == False:
                                index = v
                                found = True
                            elif int(v_gene.split('-')[1][:2]) == int(key.split('-')[1][:2]) and found == False:
                                if len(v_gene) == len(key):
                                    if int(v_gene.split('-')[2]) > int(key.split('-')[2]):
                                        index = v
                                        found = True
                                    elif int(v_gene.split('-')[2]) < int(key.split('-')[2]):
                                        index = v + 1
                                        found = True
                                elif len(v_gene) > len(key):
                                    index = v
                                    found = True
                                elif len(v_gene) < len(key):
                                    index = v + 1
                                    found = True
                        except ValueError:
                            str_v_gene.append(key)
                            break
                    if index < len(ordered_list[-1]):
                        ordered_list[-1] = ordered_list[-1][:index] + [key] + ordered_list[-1][index:]
                    else:
                        ordered_list[-1].append(key)

    vj_angle_dict = {}
    start = 0.0
    for family in ordered_list:  # Create a dictionary with start and end angles for eahc VJ gene
        for gene in family:
            for j in range(1, 7):
                vj_angle_dict[gene + '_IGHJ' + str(j)] = [start, start + 2 * np.pi * v_dict[gene][j - 1]/total]
                start += 2 * np.pi * v_dict[gene][j - 1]/total
    print('Gene positions calibrated')
    return vj_angle_dict

def isotype_to_color(isotype_list):
    color_dict = {0: '#00FF00', 1: '#98FB98', 2: '#0000CD', 3: '#0000FF', 4: '#B22222', 5: '#6495ED', 6: '#87CEEB',7: '#FF69B4', 8: '#FF0000'}

    #Set color for isotypes
    temp_array = np.array(isotype_list)
    index = np.argmax(temp_array)
    color = color_dict[index]
    return color

def depth_first_search(node, adj_matrix, visited, current_component):
    visited[node] = True
    current_component.append(node)
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
            depth_first_search(neighbor, adj_matrix, visited, current_component)

def find_connected_components(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            component = []
            depth_first_search(i, adj_matrix, visited, component)
            components.append(component)
    return components

def new_scatter_data(clonotype_dict, vj_angle_dict, label_dict_list, label_info_dict_list, label_name_list):  # Create clonotype scatter plot. Default
    node_dict = {}
    previous_dict = {}
    node_num = 0
    isotype_text_list = ['M','D','G3','G1','A1','G2','G4','E','A2']
    for key in clonotype_dict:
        isotype_index = np.argmax(np.array(clonotype_dict[key][2:11]))
        v = key.split('_')[0]
        j = key.split('_')[1]
        cdr3 = key.split('_')[2]
        label = ''
        info = ''
        group = v + '_' + j + '_' + str(len(cdr3))
        for l,label_dict in enumerate(label_dict_list):
            if group in label_dict:
                for label_cdr3 in label_dict[group]:
                    if hamming_dist(label_cdr3, cdr3) <= 0.2 * len(cdr3):
                        label = label_name_list[l] + ': '
                        info = label_info_dict_list[l][v + '_' + j + '_' + label_cdr3] + '<br>'
                        if v + '_' + j + '_' + cdr3 not in previous_dict:
                            previous_dict[v + '_' + j + '_' + cdr3] = [label + info,[l]]
                            break
                        else:
                            previous_dict[v + '_' + j + '_' + cdr3][0] += label + info
                            if l not in previous_dict[v + '_' + j + '_' + cdr3][1]:
                                previous_dict[v + '_' + j + '_' + cdr3][1].append(l)
                            break
        shm = clonotype_dict[key][11] / clonotype_dict[key][0]
        color = clonotype_dict[key][2:11]
        freq = clonotype_dict[key][1]
        angle = vj_angle_dict[v + '_' + j][0]
        clonotype_text = v + ' ' + j + ' ' + cdr3 + '<br>'
        sequence_text = 'Unique sequences: ' + str(clonotype_dict[key][0]) + '<br>'
        freq_text = 'Frequency: ' + '%.4f' % (100 * freq) + '%<br>'
        shm_text = 'Average SHM: ' + '%.1f' % (clonotype_dict[key][11] / clonotype_dict[key][0]) + '<br>'
        isotype_text = ''
        for i in range(9):
            isotype_text += isotype_text_list[i]+': '+'%.1f' % (100 * clonotype_dict[key][2+i] / freq) + '% '
        isotype_text += '<br>'
        text = ''
        temp = []
        if v + '_' + j + '_' + cdr3 in previous_dict:
            text += previous_dict[v + '_' + j + '_' + cdr3][0]
            temp = previous_dict[v + '_' + j + '_' + cdr3][1]
        text += clonotype_text + sequence_text + freq_text + shm_text + isotype_text
        if group not in node_dict:
            node_dict[group] = []
        node_dict[group].append(
            [node_num, cdr3, freq, color, shm, angle, text, [],
             '%.1f' % (clonotype_dict[key][11] / clonotype_dict[key][0]),isotype_index,clonotype_dict[key][12],clonotype_dict[key][13],temp,0])
        node_num += 1  # Create dictionary with node position and node values

    edge_normalization = 200
    for group in node_dict:  # create networks
        dist_array = np.zeros((len(node_dict[group]),len(node_dict[group])))
        for s,subject in enumerate(node_dict[group]):
            for q,query in enumerate(node_dict[group]):
                dist_array[s, q] = hamming_dist(subject[1], query[1])

        threshold = int(group.split('_')[2])*0.2
        edge_array = np.where(dist_array > threshold, 0, 1)
        component_list = find_connected_components(edge_array)
        for component in component_list:
            root_shm = 100
            root_bias_shm = 0
            root_bias_angle = 0
            root_cdr3 = ''
            cdr3_list = []
            for node in component:
                cdr3_list.append(node_dict[group][node][1])
                if node_dict[group][node][10] < root_shm:
                    root_cdr3 = node_dict[group][node][1]
                    root_shm = node_dict[group][node][4]
                    root_bias_shm = node_dict[group][node][10]
                    root_bias_angle = node_dict[group][node][11]
            min_kidera_dist = 0
            max_kidera_dist = 0
            for node in component:
                dist = kidera_dist(root_cdr3,node_dict[group][node][1])
                if dist > max_kidera_dist:
                    max_kidera_dist = dist
                if dist < min_kidera_dist:
                    min_kidera_dist = dist

            if root_bias_angle + max_kidera_dist/edge_normalization > 1:
                root_bias_angle = 1 - max_kidera_dist/edge_normalization
            if root_bias_angle + min_kidera_dist/edge_normalization < 0:
                root_bias_angle = 0 - min_kidera_dist/edge_normalization

            for node in component:
                node_dict[group][node][10] = root_bias_shm + 0.2*node_dict[group][node][10]
                dist = kidera_dist(root_cdr3, node_dict[group][node][1])
                node_dict[group][node][11] = root_bias_angle+dist/edge_normalization
                node_dict[group][node][7] = cdr3_list
                node_dict[group][node][13] = len(component)

    x_val = []
    y_val = []
    color_val = []
    size_val = []
    isotype_val = []
    label_val = []
    binder_val = []
    network_val = []
    true_shm_val = []
    bcr_sort_array = []
    isotype_dict = {0:'M',1:'M',2:'G',3:'G',4:'A',5:'G',6:'G',7:'A',8:'A'}
    for group in node_dict:
        v = group.split('_')[0]
        j = group.split('_')[1]
        for clonotype in node_dict[group]:
            radius = np.log10(clonotype[4] + 2 + 2 * clonotype[10])
            angle = vj_angle_dict[v + '_' + j][0] + clonotype[11]*(vj_angle_dict[v + '_' + j][1] - vj_angle_dict[v + '_' + j][0])
            x_val.append(radius*np.cos(angle))
            y_val.append(radius*np.sin(angle))
            color_val.append(isotype_to_color(clonotype[3]))
            size_val.append(min(1000000 * clonotype[2],1000000))  # Size offset for frequency, change if necessary
            isotype_val.append(clonotype[9])
            true_shm_val.append(clonotype[4])
            text = clonotype[6] + 'Adjacent nodes: ' + str(len(clonotype[7]))
            for cdr3 in clonotype[7]:
                text += '<br>' + cdr3
            label_val.append(text)
            binder_val.append(clonotype[12])
            network_val.append(clonotype[13])
            bcr_sort_array.append([clonotype[1],v,j,isotype_dict[clonotype[9]]])

    data_in = pd.DataFrame(bcr_sort_array, columns=['cdr3_aa','V_gene','J_gene','isotype'])
    DatasetBCR = DatasetBCRSORT(data_in=data_in)
    sequence_padder = partial(sequence_padding, mode='predict')
    DataLoaderBCR = DataLoader(DatasetBCR, batch_size=1024, num_workers=0, pin_memory=True, drop_last=False,
                               shuffle=False, collate_fn=sequence_padder)

    # load model
    model = load_model()
    model_path = '../BCR-SORT/model_weight/model_wt.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    device = 'cpu'
    model.to(device)
    model.eval()

    # setup output vars
    encoding_dict = generate_dict()
    ntest = len(DataLoaderBCR.dataset)
    num_class = 3
    pred = np.zeros((ntest, num_class))
    col_list = ['cdr3_aa', 'V_gene', 'J_gene', 'isotype', 'pred', 'p_naive', 'p_memory', 'p_diff']
    data_out = {}
    for col in col_list:
        data_out[col] = []

    # prediction
    dim_start = 0
    for i, dataloader in enumerate(DataLoaderBCR, 0):
        inputs, lengths, _ = dataloader
        sequence, vgene, jgene, isotype = inputs
        sequence = sequence.type(torch.cuda.LongTensor).to(device)
        vgene = vgene.type(torch.cuda.LongTensor).to(device)
        jgene = jgene.type(torch.cuda.LongTensor).to(device)
        isotype = isotype.type(torch.cuda.LongTensor).to(device)

        outputs, _, _ = model(sequence, vgene, jgene, isotype, lengths, False)

        # collect outputs
        dim_end = dim_start + outputs.shape[0]
        pred[dim_start:dim_end, ] = outputs.cpu().detach().numpy()

        sequence_out = decode_sequence(sequence.cpu().detach().tolist())
        vgene = vgene.squeeze().cpu().detach().tolist()
        jgene = jgene.squeeze().cpu().detach().tolist()
        isotype = isotype.squeeze().cpu().detach().tolist()

        if not isinstance(vgene, list):
            vgene = [vgene]
            jgene = [jgene]
            isotype = [isotype]

        vgene_out = [decode_categorical(f, encoding_dict['V_gene']) for f in vgene]
        jgene_out = [decode_categorical(f, encoding_dict['J_gene']) for f in jgene]
        isotype_out = [decode_categorical(f, encoding_dict['isotype']) for f in isotype]

        data_out['cdr3_aa'].extend(sequence_out)
        data_out['V_gene'].extend(vgene_out)
        data_out['J_gene'].extend(jgene_out)
        data_out['isotype'].extend(isotype_out)
        dim_start = dim_end

    # save outputs
    pred_out = np.argmax(pred, axis=1).tolist()
    cell_type_val = [decode_categorical(pred, encoding_dict['label']) for pred in pred_out]

    combined = list(zip(x_val, y_val, color_val, size_val, isotype_val, label_val, binder_val,network_val,true_shm_val,cell_type_val))
    sorted_combined = sorted(combined, key=lambda x: x[3], reverse = True)
    sorted_x = [x[0] for x in sorted_combined]
    sorted_y = [x[1] for x in sorted_combined]
    sorted_color = [x[2] for x in sorted_combined]
    sorted_size = [x[3] for x in sorted_combined]
    sorted_isotype = [x[4] for x in sorted_combined]
    sorted_label = [x[5] for x in sorted_combined]
    sorted_binder = [x[6] for x in sorted_combined]
    sorted_network = [x[7] for x in sorted_combined]
    sorted_true_shm_val = [x[8] for x in sorted_combined]
    sorted_cell_type_val = [x[9] for x in sorted_combined]

    return sorted_x, sorted_y, sorted_color, sorted_size, sorted_isotype,sorted_label,sorted_binder,sorted_network,sorted_true_shm_val,sorted_cell_type_val

def new_node_and_edge(label_file_list, label_name_list, clonotype_dict, vj_angle_dict, title, file_name, metatext,total_isotypes, save_html=True,save_png=False,dpi=500,show_id=True):
    label_dict_list = []  # Add all labels and map them to clonotypes
    label_info_dict_list = []
    if label_file_list[0] != '':
        for i in range(len(label_file_list)):
            label_dict = {}
            info_dict = {}
            file_pass = open(label_file_list[i], 'r')
            if 'CoV_AbDab' in label_file_list[i] or 'Influenza' in label_file_list[i]:
                for l, line in enumerate(file_pass):
                    split_line = line.split(',')
                    list_line = split_line[:-1] + [split_line[-1][:-1]]
                    try:
                        group = list_line[1].replace(' (Human)', '') + '_' + list_line[2].replace(' (Human)', '') + '_' + str(len(list_line[3]))
                        clonotype = list_line[1].replace(' (Human)', '') + '_' + list_line[2].replace(' (Human)', '') + '_' + list_line[3]
                        if group not in label_dict:
                            label_dict[group] = []
                        label_dict[group].append(list_line[3])
                        info_dict[clonotype] = list_line[0].replace('_merged', '')
                    except IndexError:
                        continue
            else:
                v_index = 1
                j_index = 2
                cdr3_index = 3
                label_index = 0
                for l, line in enumerate(file_pass):
                    split_line = line.split(',')
                    list_line = split_line[:-1] + [split_line[-1][:-1]]
                    if 'IGH' not in line:
                        continue
                    else:
                        group = list_line[v_index] + '_' + list_line[j_index] + '_' + str(len(list_line[cdr3_index]))
                        clonotype = list_line[v_index] + '_' + list_line[j_index] + '_' + list_line[cdr3_index]
                        if group not in label_dict:
                            label_dict[group] = []
                        label_dict[group].append(list_line[cdr3_index])
                        info_dict[clonotype] = list_line[label_index]

            label_dict_list.append(label_dict.copy())
            label_info_dict_list.append(info_dict.copy())

    plt.rcParams["figure.figsize"] = (28, 28)
    fig, ax = plt.subplots(1, 1)

    v_color_dict = {'IGHV1': '#009E73', 'IGHV2': '#F0E442', 'IGHV3': '#D55E00', 'IGHV4': '#0072B2',
                    'IGHV5': '#E69F00', 'IGHV6': '#CC79A7', 'IGHV7': '#9F9F9F'}
    isotype_text_list = ['M', 'D', 'G3', 'G1', 'A1', 'G2', 'G4', 'E', 'A2']

    radius = 1.9
    width = 0.05
    square = 2.4
    gap = 0.6

    sorted_x, sorted_y, sorted_color, sorted_size, sorted_isotype, sorted_label, sorted_binder, sorted_network, sorted_true_shm_val,sorted_cell_type_val = new_scatter_data(clonotype_dict, vj_angle_dict, label_dict_list, label_info_dict_list, label_name_list)
    scatter_plot_list = []
    tooltip_list = []

    x_all = []
    y_all = []
    color_all = []
    size_all = []
    label_all = []
    for i in range(len(sorted_isotype)):
        x_all.append(sorted_x[i])
        y_all.append(sorted_y[i])
        color_all.append(sorted_color[i])
        size_all.append(sorted_size[i])
        label_all.append(sorted_label[i].replace('Adjacent nodes','Predicted cell type: '+sorted_cell_type_val[i]+'<br>Adjacent nodes'))
    scatter_plot_list.append(ax.scatter(x_all, y_all, color=color_all, s=size_all, edgecolors='black', linewidths=1, alpha=0.7))

    tooltip_list.append(PointClickInfo(scatter_plot_list[-1], labels=label_all,divid='my-info-box',
                                       edge_color='black',highlight_edge_color='red',edge_width=1,highlight_edge_width=4,
                                       css=('''
    div#my-info-box {
        font-family: consolas;
        font-size: 28pt;
        font-weight: bold;
    }''')))

    if show_id:
        label_color = 'black'
    else:
        label_color='none'

    center = ax.scatter(0, 0, color='k', s=100, edgecolors=label_color,linewidth=1)

    tooltip = PointClickInfo(center, labels=[metatext], divid='my-info-box',
                                       edge_color='black', highlight_edge_color='black', edge_width=1,
                                       highlight_edge_width=1,
                                       css=('''
            div#my-info-box {
                font-family: consolas;
                font-size: 28pt;
                font-weight: bold;
            }'''))

    limit_chord = 1.8
    limit_text = 3.6
    max_chord = 90
    vj_gene_x = []
    vj_gene_y = []
    vj_gene_label = []
    vj_color = []

    for key in vj_angle_dict:  # Plot VJ gene chord plot
        start = (360 * vj_angle_dict[key][0]) / (2 * np.pi)
        end = (360 * vj_angle_dict[key][1]) / (2 * np.pi)
        angle = 0.5 * vj_angle_dict[key][0] + 0.5 * vj_angle_dict[key][1]
        if end - start > limit_chord and end - start < max_chord:
            chord.IdeogramArc(start=start + gap, end=end - gap,
                              radius=radius, width=width, ax=ax, color=v_color_dict[key[:5]], see_through=False)
        elif end - start >= max_chord:
            num_segment = int((end - start) / max_chord)
            for i in range(num_segment):
                if start + gap + (i + 1) * max_chord < end - gap:
                    chord.IdeogramArc(start=start + gap + i * max_chord, end=start + gap + (i + 1) * max_chord,
                                      radius=radius, width=width, ax=ax, color=v_color_dict[key[:5]], see_through=False)
                else:
                    chord.IdeogramArc(start=start + gap + i * max_chord, end=end - gap,
                                      radius=radius, width=width, ax=ax, color=v_color_dict[key[:5]], see_through=False)
            if start + gap + num_segment * max_chord < end - gap:
                chord.IdeogramArc(start=start + gap + num_segment * max_chord, end=end - gap,
                                  radius=radius, width=width, ax=ax, color=v_color_dict[key[:5]], see_through=False)

        if end - start > limit_text:  # Plot text info for major VJ genes
            vj_gene_x.append(radius * 0.975 * np.cos(angle))
            vj_gene_y.append(radius * 0.975 * np.sin(angle))
            vj_color.append(v_color_dict[key[:5]])
            if show_id:
                label_color = 'w'
            else:
                label_color = v_color_dict[key[:5]]
            vj = key.split('_')[0][3:] + ' ' + key.split('_')[1][3:]
            freq = "%.1f" % (100 * (end - start) / 360)
            vj_gene_label.append(vj + '<br>Frequency: ' + freq + '%')
            plt.text(radius * 0.985 * np.cos(angle), radius * 0.985 * np.sin(angle),
                     key.split('_')[0][3:], color=label_color, font='serif', fontsize=12,fontweight='bold',
                     ha='center', va='center',rotation=-90+0.5*(start+end))
            plt.text(radius * 0.965 * np.cos(angle), radius * 0.965 * np.sin(angle),
                     key.split('_')[1][3:], color=label_color, font='serif', fontsize=12,fontweight='bold',
                     ha='center', va='center', rotation=-90 + 0.5 * (start + end))
    vj_scatter = plt.scatter(vj_gene_x, vj_gene_y, color=vj_color, s=800,edgecolor='none',linewidth=1)

    tooltip_vj = PointClickInfo(vj_scatter, labels=vj_gene_label, divid='my-info-box',
                             edge_color='none', highlight_edge_color='none', edge_width=1,
                             highlight_edge_width=1,
                             css=('''
                div#my-info-box {
                    font-family: consolas;
                    font-size: 28pt;
                    font-weight: bold;
                }'''))

    plugins.connect(fig, tooltip_vj)

    plt.axis('off')
    plt.xlim(-0.8 * square, 0.8 * square)
    plt.ylim(-0.8 * square, 0.8 * square)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    if save_png:
        plt.savefig(file_name+'.png',dpi=dpi,transparent=True)
        plt.savefig(file_name + '.eps', dpi=dpi, format='eps',transparent=True)


    for i in range(len(isotype_text_list)):
        x = []
        y = []
        color = []
        size = []
        label = []
        for j,isotype in enumerate(sorted_isotype):
            if isotype == i:
                x.append(sorted_x[j])
                y.append(sorted_y[j])
                color.append(sorted_color[j])
                size.append(sorted_size[j])
                label.append(sorted_label[j].replace('Adjacent nodes','Predicted cell type: '+sorted_cell_type_val[j]+'<br>Adjacent nodes'))
        scatter_plot_list.append(ax.scatter(x, y, color=color, s=size, edgecolors='black', linewidths=1, alpha=0.7))

        tooltip_list.append(PointClickInfo(scatter_plot_list[-1], labels=label,divid='my-info-box',
                                           edge_color='black',highlight_edge_color='red',edge_width=1,highlight_edge_width=4,
                                           css=('''
        div#my-info-box {
            font-family: consolas;
            font-size: 28pt;
            font-weight: bold;
        }''')))

    network_size = [10, 20, 50, 100]
    for i in network_size:
        x = []
        y = []
        color = []
        size = []
        label = []
        for j, network in enumerate(sorted_network):
            if network >= i:
                x.append(sorted_x[j])
                y.append(sorted_y[j])
                color.append(sorted_color[j])
                size.append(sorted_size[j])
                label.append(sorted_label[j].replace('Adjacent nodes','Predicted cell type: '+sorted_cell_type_val[j]+'<br>Adjacent nodes'))
        scatter_plot_list.append(ax.scatter(x, y, color=color, s=size, edgecolors='black',linewidth=1, alpha=0.7))

        tooltip_list.append(PointClickInfo(scatter_plot_list[-1], labels=label, divid='my-info-box',
                                           edge_color='black', highlight_edge_color='red', edge_width=1,
                                           highlight_edge_width=4,
                                           css=('''
                div#my-info-box {
                    font-family: consolas;
                    font-size: 28pt;
                    font-weight: bold;
                }''')))


    for i in range(len(label_dict_list)):  # Scatter plot according to labels
        x = []
        y = []
        color = []
        size = []
        label = []
        for b,binder in enumerate(sorted_binder):
            if i in binder:
                x.append(sorted_x[b])
                y.append(sorted_y[b])
                color.append(sorted_color[b])
                size.append(sorted_size[b])
                label.append(sorted_label[b].replace('Adjacent nodes','Predicted cell type: '+sorted_cell_type_val[b]+'<br>Adjacent nodes'))
        scatter_plot_list.append(ax.scatter(x, y, color=color, s=size, edgecolors='black',linewidth=1, alpha=0.7))

        tooltip_list.append(PointClickInfo(scatter_plot_list[-1], labels=label, divid='my-info-box',
                                           edge_color='black', highlight_edge_color='red', edge_width=1,
                                           highlight_edge_width=4,
                                           css=('''
                div#my-info-box {
                    font-family: consolas;
                    font-size: 28pt;
                    font-weight: bold;
                }''')))

    cell_type = ['Naive','Memory','ASC']
    for i in cell_type:  # Scatter plot according to BCRsort labels
        x = []
        y = []
        color = []
        size = []
        label = []
        for c, cell in enumerate(sorted_cell_type_val):
            if cell == i:
                x.append(sorted_x[c])
                y.append(sorted_y[c])
                color.append(sorted_color[c])
                size.append(sorted_size[c])
                label.append(sorted_label[c].replace('Adjacent nodes','Predicted cell type: '+sorted_cell_type_val[c]+'<br>Adjacent nodes'))
        scatter_plot_list.append(ax.scatter(x, y, color=color, s=size, edgecolors='black', linewidth=1, alpha=0.7))

        tooltip_list.append(PointClickInfo(scatter_plot_list[-1], labels=label, divid='my-info-box',
                                           edge_color='black', highlight_edge_color='red', edge_width=1,
                                           highlight_edge_width=4,
                                           css=('''
                   div#my-info-box {
                       font-family: consolas;
                       font-size: 28pt;
                       font-weight: bold;
                   }''')))

    for i in range(len(tooltip_list)):
        plugins.connect(fig, tooltip_list[i])

    labels = ['Total','IgM', 'IgD', 'IgG3', 'IgG1', 'IgA1', 'IgG2', 'IgG4', 'IgE', 'IgA2'] + ['Cluster +'+str(x) for x in network_size] + label_name_list + ['Naive','Memory','ASC']
    interactive_legend = plugins.InteractiveLegendPlugin(scatter_plot_list, labels, font_size=24, alpha_unsel=0.00,
                                                         start_visible=[True] + [False] * (len(labels) - 1),
                                                         legend_offset=(-80, 220))

    plugins.connect(fig, tooltip, interactive_legend)  # Connect interactive legend plugin

    if save_html:
        mpld3.save_html(fig, file_name+'.html')

    plt.close()
    return
