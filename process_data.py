"""Process MultiWOZ dataset for dialog-act-to-generation task.
Check out the paper https://aclanthology.org/D18-1547.pdf for task description.
"""


import os
import json
import zipfile

from tqdm import tqdm


def unzip_multiwoz(src, dest):
    """Unzip MultiWOZ dataset.
    @param  src     file path to zip file
    @param  dest    directory to unzip files
    """
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dest)
        zip_ref.close()


def linearize_act(dialog_act):
    """Linearize structured data.
    @param  dialog_act      a structured data that contains
                            dialog act to linearize

    @return linearized_act  linearized act in string
    """
    # return None in case dialog_act is "No Annotation"
    if type(dialog_act) is str:
        return None

    action_str_list = []
    for act_key, act_val in dialog_act.items():
        sv_str_list = []
        for slot, value in act_val:
            sv_str_list.append(f"{slot} = {value}")
        domain, action = act_key.split('-')     # e.g. Attraction-Inform ==> Attraction, Inform
        action_str_list.append(f"{domain} {action} {' , '.join(sv_str_list)}")
    
    linearized_act = ' | '.join(action_str_list)
    return linearized_act
            

def process_data(data_dir, dialog_ids):
    """Process MultiWOZ dataset.
    @param  data_dir    directory that contains data.json and dialog_acts.json
    @param  dialog_ids  set of dialog ids for either train/dev/test

    @return proc_data   a list of linearized act and utterance pairs
    """
    # load dialogs and system_acts
    with open(os.path.join(data_dir, 'data.json'), 'r') as json_file:
        dialogs = json.load(json_file)
    with open(os.path.join(data_dir, 'dialogue_acts.json'), 'r') as json_file:
        system_acts = json.load(json_file)

    # make sure dialog ids are shared
    assert set(dialogs.keys()) == set([dialog_id + '.json' for dialog_id in system_acts.keys()])

    # process data
    proc_data = []
    prog_bar = tqdm(dialog_ids)
    for dialog_id in prog_bar:
        pairs = construct_dialog_pairs(dialogs, system_acts, dialog_id)
        proc_data.extend(pairs)
        prog_bar.set_description(f"processing: {dialog_id}")

    return proc_data


def construct_dialog_pairs(dialogs, system_acts, dialog_id):
    """Construct linearized act and utterance pairs from a dialog.
    @param  dialogs         dictionary form of data.json
    @param  system_acts     dictionary form of dialogue_acts.json
    @param  dialog_id       dialog id e.g. SNG0760

    @return dialog_pairs    paired linearized acts and utterances from a dialog
    """
    dialog = dialogs[dialog_id + '.json']
    sys_act = system_acts[dialog_id]

    dialog_pairs = []

    for sys_turn_idx in map(lambda k: int(k), sys_act.keys()):
        # there are garbage acts or missing acts
        # skip if garbage act attempts to access invalid turn
        if len(dialog['log']) < sys_turn_idx * 2:
            continue

        pair = construct_utterance_pair(dialog, sys_act, sys_turn_idx)

        # in case dialog_act is "No Annotation"
        if pair[0] is not None:
            dialog_pairs.append(pair)
    
    return dialog_pairs


def construct_utterance_pair(dialog, system_act, turn_idx):
    """Construct an input and output pair
    @param  dialog          dictionary that contains user/system utterances
    @param  system_act      dictionary holding system acts
    @param  turn_idx        turn index in a dialog
    
    @return linearized_act  linearized act in string
    @return utt             utterance
    """

    utt = dialog['log'][turn_idx*2 - 1]['text']
    act = system_act[str(turn_idx)]

    linearized_act = linearize_act(act)

    return linearized_act, utt



def split_dialog_ids(data_dir):
    """Split dialog ids into train, dev, and test set.
    @param  data_dir    directory that contains data.json and dialog_acts.json

    @return train_ids   dialog ids for train set
    @return dev_ids     dialog ids for dev set
    @return test_ids    dialog ids for test set
    """
    # load dialogs
    with open(os.path.join(data_dir, 'data.json'), 'r') as json_file:
        dialogs = json.load(json_file)
    
    dialog_ids = set(map(lambda x: os.path.splitext(x)[0], dialogs.keys()))
    # dialog_ids = set(dialogs.keys())

    with open('data/MULTIWOZ2 2/valListFile.json', 'r') as f:
        dev_ids = set(map(lambda x: os.path.splitext(x)[0], f.read().splitlines()))
    
    with open('data/MULTIWOZ2 2/testListFile.json', 'r') as f:
        test_ids = set(map(lambda x: os.path.splitext(x)[0], f.read().splitlines()))

    assert not set.intersection(dev_ids, test_ids), "There shouldn't be overlapping dialog ids between test and dev"
    
    train_ids = dialog_ids.difference(set.union(dev_ids, test_ids))
    return train_ids, dev_ids, test_ids
    


def main():
    # prepare raw dataset
    unzip_multiwoz('multiwoz/data/MultiWOZ_2.0.zip', 'data/')
    print("Unzip done.")

    # split train/dev/test dialog ids
    train_ids,dev_ids, test_ids = split_dialog_ids('data/MULTIWOZ2 2')

    # process data
    train_proc_data = process_data('data/MULTIWOZ2 2', train_ids)
    dev_proc_data = process_data('data/MULTIWOZ2 2', dev_ids)
    test_proc_data = process_data('data/MULTIWOZ2 2', test_ids)
    print("Data process finished.")

    # save processed data
    train_proc_data_dict = [{'act': act, 'utterance': utt} for act, utt in train_proc_data]
    dev_proc_data_dict = [{'act': act, 'utterance': utt} for act, utt in dev_proc_data]
    test_proc_data_dict = [{'act': act, 'utterance': utt} for act, utt in test_proc_data]
    with open('data/train_processed.json', 'w') as json_file:
        json.dump(train_proc_data_dict, json_file, indent=2)
    with open('data/dev_processed.json', 'w') as json_file:
        json.dump(dev_proc_data_dict, json_file, indent=2)
    with open('data/test_processed.json', 'w') as json_file:
        json.dump(test_proc_data_dict, json_file, indent=2)
    print("Processed data is saved at data/processed.json")
    

        

if __name__ == '__main__':
    main()
