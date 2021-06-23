from subprocess import Popen, PIPE
from shlex import split

def getTS(
    dataset_repertory, 
    output_file="timestamps.txt",
    frame_interval = 2.89 #ms 
    ): 
    
    # get number of files in input dataset
    p1 = Popen(split("ls "+dataset_repertory), stdout=PIPE)
    p2 = Popen(split("wc -l"), stdin=p1.stdout, stdout=PIPE)
    out, err = p2.communicate()
    nb_files = int(out.strip().decode('ascii'))

    # create file 
    ts_file = open(output_file, "w")
    ts = 0
    while ts < nb_files-2:
        ts_file.write(str(ts*frame_interval)[::-1].zfill(9)[::-1]+"\n")
        ts += 1
    ts_file.write(str(ts*frame_interval)[::-1].zfill(9)[::-1])
    ts_file.close() 