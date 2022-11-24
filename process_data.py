"""Process MultiWOZ dataset for dialog-act-to-generation task.
Check out the paper https://aclanthology.org/D18-1547.pdf for task description.
"""


import os
import json
import zipfile



def unzip_multiwoz(src, dest):
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dest)
        zip_ref.close()


def linearize_act(dialog_act):
    """Linearize structured data"""

    pass

def process_data(data_dir):
    # load dialogs and system_acts
    with open(os.path.join(data_dir, 'data.json'), 'r') as json_file:
        dialogs = json.load(json_file)
    with open(os.path.join(data_dir, 'dialogue_acts.json'), 'r') as json_file:
        system_acts = json.load(json_file)

    # make sure dialog ids are shared
    assert set(dialogs.keys()) == set([dialog_id + '.json' for dialog_id in system_acts.keys()])

    # process data
    proc_data = []
    for dialog_id in system_acts:
        pairs = construct_dialog_pairs(dialogs, system_acts, dialog_id)
        proc_data.extend(pairs)

    return proc_data


def construct_dialog_pairs(dialogs, system_acts, dialog_id):
    dialog = dialogs[dialog_id + '.json']
    sys_act = system_acts[dialog_id]

    dialog_pairs = []
    
    # sanity check: act sometimes has garbage annotation in the end
    assert len(dialog['log']) <= len(sys_act) * 2, f"Sanity check failed: {dialog_id}"

    for sys_turn_idx in range(len(dialog['log']) // 3):
        pair = construct_utterance_pair(dialog, sys_act, sys_turn_idx)
        dialog_pairs.append(pair)
    
    return dialog_pairs



def construct_utterance_pair(dialog, system_act, turn_idx):
    """Construct input and output pair (linearized_act, utterance)"""

    utt = dialog['log'][turn_idx*2 + 1]['text']
    act = system_act[str(turn_idx + 1)]

    linearized_act = linearize_act(act)

    return linearized_act, utt


def main():
    # unzip_multiwoz
    # process_data
    pass

        

if __name__ == '__main__':
    # unzip_multiwoz('multiwoz/data/MultiWOZ_2.0.zip', 'data/')
    # print("Unzip done.")

    # test
    process_data('data\MULTIWOZ2 2')
