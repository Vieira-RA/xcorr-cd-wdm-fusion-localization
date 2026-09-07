#!/usr/bin/env python
"""
Compare readburstpm vs sequential readpm for the 8 PM1000 Stokes registers,
to determine the correct argument mapping and return type.
"""
import Python_USB
import matlab

DESCR = "PM1000-100M-XL-FA-N20-D 82"
REGS = [512+10, 512+11, 512+12, 512+13, 512+14, 512+15, 512+16, 512+17]


def to_list(data):
    """Try to convert a MATLAB array return into a list of Python ints."""
    try:
        return [int(v) for v in data]
    except TypeError:
        return [int(data)]


def try_burst(pm, label, rdaddr, addrstart, addrstop, wraddr):
    print("---", label, "---")
    data, okb = pm.readburstpm(rdaddr, addrstart, addrstop, wraddr, nargout=2)
    print("ok:", okb, "type:", type(data).__name__)
    try:
        print("values:", to_list(data))
    except Exception as e:
        print("convert failed:", e, "repr:", repr(data))
    return okb


def main():
    pm = Python_USB.initialize()
    ok, diagnosis, handle, in_descr, out_descr = pm.initpm(
        matlab.uint16([3]), DESCR, nargout=5)
    if not ok:
        print("initpm FAILED:", diagnosis)
        pm.terminate()
        return
    print("PM1000 initialized.")

    # 1) sequential reads (known-good reference)
    seq = []
    for addr in REGS:
        val, _ok = pm.readpm(matlab.double([addr], size=(1, 1)), nargout=2)
        seq.append(int(val))
    print("sequential (phys 522..529):", seq)
    print("sequential (offsets 10..17):", [s for s in seq])

    def u16(v):
        return matlab.uint16([v], size=(1, 1))

    # 2) sample's exact values (offsets 3..13)
    try_burst(pm, "sample 3..13", u16(3), u16(3), u16(13), u16(3))

    # 3) offsets 10..17 (raw register offsets, no 512 base)
    try_burst(pm, "offsets 10..17", u16(10), u16(10), u16(17), u16(10))

    # 4) physical 522..529 as double (matching readpm's double convention)
    def dbl(v):
        return matlab.double([v], size=(1, 1))
    try_burst(pm, "phys 522..529 (double)", dbl(522), dbl(522), dbl(529), dbl(522))

    pm.closepm()
    pm.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
