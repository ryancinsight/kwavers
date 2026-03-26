import inspect
from kwave.utils.signals import tone_burst
with open("tb_source.txt", "w") as f:
    f.write(inspect.getsource(tone_burst))
