# Set arguments as global vars
percentile = 0.5
warmup = 32
no_txt = False
no_json = False
no_decode = False

retain = []

def set_thresh_ppl_args(p, w, t, j):
    global percentile, warmup, no_txt, no_json
    percentile = p
    warmup = int(w)
    no_txt = t
    no_json = j

def set_thresh_tasks_args(p, w, t, d):
    global percentile, warmup, no_txt, no_decode
    percentile = p
    warmup = int(w)
    no_txt = t
    no_decode = d

def update_retain_data(retain_sample):
    retain.append(retain_sample)
