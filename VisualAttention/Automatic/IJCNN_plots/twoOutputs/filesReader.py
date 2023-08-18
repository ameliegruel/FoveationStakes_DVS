import os
from datetime import timedelta

def read_file(file_path):
    
    # read data in info.csv produced during saliency detection
    data = []
    keys = None

    info_file = open( 
        os.path.join(file_path, 'info.csv'), "r"
    )
    for line in info_file.readlines():
        line = line.split(';')
        line.remove('\n')
        if len(line) == 0:
            pass
        elif line[0] == 'original dataset':
            sample_file = line[1]
        elif keys == None:
            keys = {k:i for i,k in enumerate(line)}
        else: 
            dico = {
                k: int(line[i]) for k,i in keys.items() if k not in ['time', 'sample name']
            }
            dico['sample name'] = line[keys['sample name']]            
            
            time = line[keys['time']].split(":")
            if len(time) > 1:
                time = timedelta(
                    hours=float(time[0]),
                    minutes=float(time[1]),
                    seconds=float(time[2])
                ).total_seconds()
            else:
                time = float(time[0])
            dico['time'] = time

            data.append(dico)
    info_file.close()

    # read data on samples in references_XXX.csv and probabilities_XXX.csv files produced during combinated samples generation
    references = {}
    statistics = {}
    labels = ['references', 'statistics']
    keys = None
    sample_rep = os.path.join(
        '/'.join(file_path.split('/')[:-2]),
        sample_file
    )
    sample_files = {k: os.path.join(sample_rep,f) for k in labels for f in os.listdir(sample_rep) if f.startswith(k)}
    
    # fill references dictionary
    ref_file = open(sample_files['references'], 'r')
    for line in ref_file.readlines():
        line = line.split(';')
        line.remove('\n')
        if len(line) == 0:
            pass
        elif keys == None:
            keys = [[i,k] for i,k in enumerate(line) if k != 'file name']
        else: 
            references[line[0]] = { 
                k: line[i] for i,k in keys
            }
    ref_file.close()

    # fill statistics dictionary
    keys = None
    stat_file = open(sample_files['statistics'], 'r')
    for line in stat_file.readlines():
        line = line.split(';')
        line.remove('\n')
        if len(line) == 0:
            pass
        elif keys == None:
            keys = [[i,k] for i,k in enumerate(line) if k != 'sample name']
        else: 
            statistics[line[0]] = { 
                k: float(line[i]) for i,k in keys
            }
    ref_file.close()

    return data, references, statistics