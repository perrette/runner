def parse_val(s):
    " string to int, float, str "
    try:
        val = int(s)
    except:
        try:
            val = float(s)
        except:
            val = s
    return val
