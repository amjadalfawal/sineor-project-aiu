#Author : amjad alfawal
import glob
import os.path
import os
import numpy as np
import h5py
AMINO_ACIDE_LENGTH = 2000

def process_text_based_data(force_overwrite=True):
    print("Starting pre-processing of text_based data.............................................................................")
    input_files_data_set = glob.glob("data/text_based/*")
    input_files_data_set_filtered = filter_input_files_data_set(input_files_data_set)
    for file_path in input_files_data_set_filtered:
        filename = file_path.split('/')[-1]
        preprocessed_file_name = "data/preprocessed/"+'sample'+".hdf5"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_overwrite:
                print("force_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name)
    print("Completed pre-processing-process.")


def process_file(input_file, output_file):
    print("Processing text_based data file", input_file)

    # create output file
    f = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocaton = 0
    temp_dest1 = f.create_dataset('primary',(current_buffer_size,AMINO_ACIDE_LENGTH),maxshape=(None,AMINO_ACIDE_LENGTH),dtype='int32')
    temp_dest2 = f.create_dataset('tertiary',(current_buffer_size,9,AMINO_ACIDE_LENGTH),maxshape=(None,9, AMINO_ACIDE_LENGTH),dtype='float')
    temp_dest3 = f.create_dataset('mask',(current_buffer_size,AMINO_ACIDE_LENGTH),maxshape=(None,AMINO_ACIDE_LENGTH),dtype='float')

    input_f_pointer = open("data/" + input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein = read_strcture_protine(input_f_pointer)
        if next_protein is None:
            break
        if current_buffer_allocaton >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            temp_dest1.resize((current_buffer_size,AMINO_ACIDE_LENGTH))
            temp_dest2.resize((current_buffer_size,9 ,AMINO_ACIDE_LENGTH))
            temp_dest3.resize((current_buffer_size,AMINO_ACIDE_LENGTH))


        sequence_length = len(next_protein['primary'])

        if sequence_length > AMINO_ACIDE_LENGTH:
            print("skip seq Length:", sequence_length)
            continue

        primary_pad = np.zeros(AMINO_ACIDE_LENGTH)
        tertiary_pad = np.zeros((9, AMINO_ACIDE_LENGTH))
        mask_pad = np.zeros(AMINO_ACIDE_LENGTH)

        primary_pad[:sequence_length] = next_protein['primary']
        tertiary_pad[:,:sequence_length] = np.array(next_protein['tertiary']).reshape((9,sequence_length))
        mask_pad[:sequence_length] = next_protein['mask']

        temp_dest1[current_buffer_allocaton] = primary_pad
        temp_dest2[current_buffer_allocaton] = tertiary_pad
        temp_dest3[current_buffer_allocaton] = mask_pad
        print(primary_pad)
        print(tertiary_pad)
        print(mask_pad)
        current_buffer_allocaton += 1
        

    print("number of Wrote output", current_buffer_allocaton, "proteins in file", output_file)

def filter_input_files_data_set(input_files_data_set):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files_data_set))



def read_strcture_protine(f_pointer):

        dict_ = {}
        _seq_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,
                    'Y': 20}
        _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
        _mask_dict = {'-': 0, '+': 1}

        while True:
            next_line = f_pointer.readline()
            if next_line == '[ID]\n':
                id_ = f_pointer.readline()[:-1]
                dict_.update({'id': id_})
            elif next_line == '[PRIMARY]\n':
                primary = list([_seq_dict[aa] for aa in f_pointer.readline()[:-1]])
                dict_.update({'primary': primary})
            elif next_line == '[EVOLUTIONARY]\n':
                evolutionary = []
                for residue in range(21): evolutionary.append(
                    [float(step) for step in f_pointer.readline().split()])
                dict_.update({'evolutionary': evolutionary})
            elif next_line == '[SECONDARY]\n':
                secondary = list([_dssp_dict[dssp] for dssp in f_pointer.readline()[:-1]])
                dict_.update({'secondary': secondary})
            elif next_line == '[TERTIARY]\n':
                tertiary = []
                # 3 dimension
                for axis in range(3): tertiary.append(
                    [float(coord) for coord in f_pointer.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif next_line == '[MASK]\n':
                mask = list([_mask_dict[aa] for aa in f_pointer.readline()[:-1]])
                dict_.update({'mask': mask})
            elif next_line == '\n':
                return dict_
            elif next_line == '':
                return None
