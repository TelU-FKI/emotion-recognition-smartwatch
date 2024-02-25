# Importing necessary libraries
import numpy as np
import argparse
import sys
import glob

# Main function
def main():
    '''
    Run as:
    python get_walking_data.py user_study_encoding.csv input_directory output_directory

    Example:
    python get_walking_data.py user_study_encoding.csv raw_data walking_data

    Takes a csv file of start-stop end times for each participant,
    and generates a csv file with only the walking times.
    The output file is formatted as follows:
    condition,emotion,walking_data

    where condition is one of mo, mu, and mw
    and emotions are sad, neutral, happy
    '''

    # Mapping of emotions and their codes
    EMOTIONS = {'s': -1, 'n': 0, 'h': 1}
    R_EMOTIONS = {-1:'s', 0:'n', 1:'h'}
    # Mapping of conditions and their codes
    CONDITIONS = {'mo': 0, 'mu': 1, 'mw': 2}

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="frame extraction")
    parser.add_argument("encoding", type=str, help="file containing participant list")
    parser.add_argument("input_dir", type=str, help="directory containing participant data")
    parser.add_argument('output_dir', type=str, help='output directory')

    args = parser.parse_args()
    input_dir = args.input_dir.strip('/') + '/'
    # .strip('/') + '/' itu untuk menghapus / di awal dan akhir string dan menambahkan / di akhir string
    print("input: ", args.input_dir)
    print("strip: ", args.input_dir.strip('/'))
    print("input dir: ", input_dir)
    # .strip itu apa? itu untuk menghapus spasi di awal dan akhir string
    output_dir = args.output_dir.strip('/') + '/'
    # .strip('/') itu untuk menghapus / di awal dan akhir string

    # Reading participant id and condition code from encoding file
    encoding = np.genfromtxt(args.encoding, delimiter=',', dtype=str, skip_header=1)
    # encoding[:,0] itu untuk ambil semua baris, dan kolom pertama
    # encoding[:,1] itu untuk ambil semua baris, dan kolom kedua
    participants = [p.lower() for p in encoding[:,0]] # participant ids, jadi dibuat menjadi menjadi huruf kecil semua
    conditions = [p.lower() for p in encoding[:,1]]

    for i, pid in enumerate(participants):
        # enumerate itu untuk mengembalikan indeks dan value dari suatu list
        
        condition, emotions = conditions[i].split('-')
        # conditions[i].split('-') itu untuk memisahkan antara condition dan emotions, dengan pemisahnya adalah '-' karena ini untuk kasus seperti "Mo-SNH"
        
        # Finding file matching participant id
        fname = glob.glob(input_dir + pid + '_*')
        if len(fname) > 1: # jika panjang dari fname lebih dari 1
            print('^^^^ more than one file matched %s ' % pid)
            return
        fname = fname[0] # mengambil file pertama
        short_name = fname.split('/')[-1] # mengambil nama file saja
        print('fname: ', fname)
        print('fname split: ', fname.split('/'))
        # bagaimana bisa split('/') itu menghasilkan nama file saja?
        print('processing ', short_name) # print nama file

        # Reading data from file
        time = np.genfromtxt(fname, delimiter=',', dtype=str, usecols=(0), skip_header=2, skip_footer=1)
        time = [':'.join([frag.zfill(2) for frag in t.split(':')[:-1]]) for t in time.tolist()] # mengambil waktu saja
        # .zfill itu untuk menambahkan 0 di depan string untuk waktu seperti 1:2:3 menjadi 01:02:03
        # .split(':')[:-1] itu untuk mengambil waktu saja, bukan detiknya

        # Reading accelerometer, gyroscope, and heart rate data
        data = np.genfromtxt(fname, delimiter=',', usecols=(1,2,3,7,8,9,10), skip_header=2, skip_footer=1)

        # Getting start-stop times
        start_stop_time = [t.replace('.', ':') for t in encoding[i, -6:].tolist()]

        # Encoding start-stop encoding[i, -6:] itu untuk mengambil 6 kolom terakhir dari baris ke-i

        indexes = []
        for time_ in start_stop_time:
            if time_ in time:
                index = time.index(time_)
                indexes.append(index)
            else:
                print('invalid index ', short_name, start_stop_time)
                break

        # Checking for valid indexes
        if len(indexes) != 6:
            print('missing an index ', short_name)
            continue

        invalid = False
        for i in range(len(indexes)-1):
            if indexes[i] >= indexes[i+1]: # jika index ke-i lebih besar atau sama dengan index ke-i+1
                invalid = True
                break

        if invalid:
            print('invalid index ', indexes)
            continue
            
        # Compiling the rows between each start-stop 
        document = []

        id_ = 0
        emotion_col = [] # List of emotion columns
        # For all sets of walking time
        for set_id, i in enumerate(range(0, len(indexes), 2)): # ini untuk apa? untuk mengambil index dari indexes
            # apa itu set_id? itu adalah index dari indexes
            # bedanya set_id dan i? set_id itu index dari indexes, i itu index dari range(0, len(indexes), 2)
            start, end = indexes[i], indexes[i+1] # start dan end adalah index ke-i dan ke-i+1
            document.extend(data[start:end])
            # .extend itu untuk menambahkan data[start:end] ke document
            # Compiling emotion codes as list
            emo_code = EMOTIONS.get(emotions[set_id]) 
            # .get itu untuk mengambil value dari suatu key
            emotion_col.extend([emo_code] * (end-start))
            # emotion_col itu untuk menambahkan emo_code sebanyak (end-start) kali

        # Creating condition column, same for all
        condition_col = [CONDITIONS[condition]] * len(document)

        # Adding frame ids as first column
        document = np.column_stack((condition_col, emotion_col, document))
        
        # Saving the document
        np.savetxt('{}{}_{}'.format(output_dir, condition, short_name), 
                        document, delimiter=',', 
                        header='condition,emotion,data', fmt='%s')

if __name__ == "__main__":
    main()
