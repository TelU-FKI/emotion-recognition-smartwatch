import argparse
import math
import numpy as np

from collections import OrderedDict
from scipy import stats
from scipy import signal


def filter(data, fs): 
    # Third order median filter to remove noise from accelerometer data
    print("Data shape: " + data.shape)
    print("data 1: " + data[:,0])
    print("data 2: " + data[:,1])
    print("data 3: " + data[:,2])
    total_acc_x = signal.medfilt(data[:,0], 3) # 3rd order median filter
    total_acc_y = signal.medfilt(data[:,1], 3) # 3rd order median filter
    total_acc_z = signal.medfilt(data[:,2], 3) # 3rd order median filter
    print("total_acc_x: " + total_acc_x)
    print("total_acc_y: " + total_acc_y)
    print("total_acc_z: " + total_acc_z)
    data[:, 0] = total_acc_x # replace the first column with the filtered data
    data[:, 1] = total_acc_y
    data[:, 2] = total_acc_z
    print("data 1-1: " + data[:,0])
    print("data 2-1: " + data[:,1])
    print("data 3-1: " + data[:,2])
    return data


def angle_between_vectors(a, b):
    # Calculate the angle between two vectors using dot product and cross product
    print("a: " + a)
    print("b: " + b)
    dot = np.dot(a, b) # np.dot is the dot product of two arrays
    # np.dot(a, b) = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    print("dot: " + dot)
    cp = np.cross(a, b) # np.cross is the cross product of two arrays
    # np.cross(a, b) = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    print("cp: " + cp)
    cp_norm = np.sqrt(np.sum(cp * cp)) # np.sqrt is the square root of an array, np.sum is the sum of an array
    print("cp_norm: " + cp_norm)
    angle = math.atan2(cp_norm, dot) # math.atan2 is the arctangent of two arrays
    print("angle: " + angle)
    return angle


def get_feature_vector(data):
    # Extract various statistical features from the given data column
    feature_functions = [
                       np.mean,  # 1. Mean
                       np.amax,  # 2. Maximum
                       np.amin,  # 3. Minimum
                       np.std,   # 4. Standard Deviation
                       lambda d: np.sum(d**2)/d.shape[0],  # 5. Energy
                       stats.kurtosis,  # 6. Kurtosis, itu adalah ukuran seberapa tajam puncak distribusi data
                       stats.skew,  # 7. Skewness, itu adalah ukuran seberapa simetris distribusi data
                       lambda d: np.sqrt(np.mean(np.square(d))),  # 8. Root Mean Square
                       lambda d: np.sqrt(np.sum(np.square(d))),  # 9. Root Sum of Squares
                       np.sum,  # 10. Area
                       lambda d: np.sum(np.abs(d)),  # 11. Absolute Area
                       lambda d: np.mean(np.abs(d)),  # 12. Absolute Mean
                       lambda d: np.amax(d)-np.amin(d),  # 13. Range
                       lambda d: np.percentile(d, 25),  # 14. 1st Quartile
                       lambda d: np.percentile(d, 50),  # 15. 2nd Quartile (Median)
                       lambda d: np.percentile(d, 75),  # 16. 3rd Quartile
                       lambda d: np.median(np.abs(d - np.median(d)))]  # 17. Median Absolute Deviation

    features = [f(data) for f in feature_functions]

    return features


def extract_features(window):
    # Extract features from the given window of data
    features = []
    heart_rate = window[:, -1] # last column is heart rate, jadi -1 itu untuk mengambil kolom terakhir
    window_no_hr = window[:, :-1] # remove heart rate column
    for column in window_no_hr.T:  # iterate over each column
        features.extend(get_feature_vector(column))

    # Calculate additional features related to acceleration and angles
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    # 51 + 3

    vector = np.array([np.mean(x), np.mean(y), np.mean(z)])
    angle_wrt_xaxis = angle_between_vectors(vector, np.array([1, 0, 0]))
    angle_wrt_yaxis = angle_between_vectors(vector, np.array([0, 1, 0]))
    angle_wrt_zaxis = angle_between_vectors(vector, np.array([0, 0, 1]))
    features.extend([angle_wrt_xaxis, angle_wrt_yaxis, angle_wrt_zaxis])

    ## magnitude - std - 1
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features.append(np.std(magnitude))

    # (17*3) + (17*3) + 3 + 1 + 1 (hr) = 107
    # + y label = 108

    features.append(heart_rate[0]) # apa isi heart_rate[0]? heart_rate[0] itu adalah isi dari baris pertama kolom terakhir, jadi nanti di akhir hr-nya disatukan

    return features


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", metavar='file', type=str, nargs='+', help="file containing acc data")
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument("-w", help="window size (whole seconds)", type=float, default=1)
    parser.add_argument("--overlap", help="overlap (percent, i.e. 0, 0.5, 0.8)", type=float, default=0.5)
    parser.add_argument("-d", "--delimiter", type=str, help="delimiter used in file, default is , (csv)", default = ',')
    args = parser.parse_args()

    window_size_sec = args.w
    overlap = args.overlap
    input_files = args.input_files
    output_dir = args.output_dir.strip('/') + '/'
    delimiter = args.delimiter

    FREQ_RATE = 24.0
    
    window_size = int(window_size_sec * FREQ_RATE)
    step = int(window_size * (1.-overlap))

    for fname in input_files:
        short_name = fname.split('/')[-1]
        print('processing ', short_name)
        condition_emotion = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=(0,1))
        emotions = map(int, condition_emotion[:,1].tolist())

        data = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=range(2, 9))

        # get emotions from second column
        emotion_ids = list(OrderedDict.fromkeys(emotions))
        emo_0 = emotions.index(emotion_ids[0])
        emo_1 = emotions.index(emotion_ids[1])
        emo_2 = emotions.index(emotion_ids[2])
        frames = [(emo_0, emo_1), (emo_1, emo_2), (emo_2, len(emotions))]

        features = []

        for (fstart, fend), label in zip(frames, emotion_ids):

            # filter data within start-end time, except heart rate
            data[fstart:fend,:-1] = filter(data[fstart:fend,:-1], FREQ_RATE)
            # extract consecutive windows
            i = fstart
            while i+window_size < fend:
                window = data[i:i+window_size]

                f_vector = extract_features(window)
                f_vector.append(label)
                features.append(f_vector)
                i += step

        features = np.array(features)

        filename = 'features_{}'.format(short_name)
        print('\tSaving file {}...'.format(filename))
        np.savetxt(output_dir + filename, features, fmt='%f', delimiter=',')
        print('\tfeatures: ', features.shape)


if __name__ == "__main__":
    main()
