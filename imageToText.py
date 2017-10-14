import numpy as np

def imageToText(preds):
    # print (preds)
    preds_modify = np.where(preds != 4, np.zeros(preds.shape,dtype=np.int), np.ones(preds.shape,dtype=np.int))
    parts_name = ['left', 'front', 'right']
    part_idx = [int(round(preds.shape[2]/3)),int(round(preds.shape[2]*2/3))]
    height_idx = int(round(preds.shape[1]*2/3))
    parts = []
    parts.append(preds_modify[:,height_idx:,:part_idx[0],:])
    parts.append(preds_modify[:,height_idx:, part_idx[0]:part_idx[1], :])
    parts.append(preds_modify[:,height_idx:, part_idx[1]:, :])

    def isobstacle(part):
        percentage_all = np.sum(part)/float(part.size)
        center = part[:,part.shape[1]/4:,part.shape[2]/4:part.shape[2]*3/4,:]
        percentage_center = np.sum(center)/float(center.size)
        print (percentage_all, percentage_center)
        if percentage_all > 0.9 and percentage_center>0.99:
            return 0
        else:
            return 1
    out = ''

    obstacle_count = 0
    blocked = [0,0,0]
    for i in range(len(parts)):
        obstacle = isobstacle(parts[i])
        if obstacle == 1:
            if obstacle_count >= 1:
                out += ' and '
            out += ('Watch ' + parts_name[i]) if out is '' else parts_name[i]
            obstacle_count += 1
            blocked[i] = 1
    if obstacle_count is 3:
        out = 'All blocked.'

    out = 'All clear!' if out is '' else out
    return blocked, out
