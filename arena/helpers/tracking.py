import re
def get_timestamp(key):
    l = re.findall('_t(\d+)', key)
    assert len(l) == 1
    return int(l[0])