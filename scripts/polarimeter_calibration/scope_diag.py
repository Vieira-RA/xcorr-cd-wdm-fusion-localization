import vxi11

s = vxi11.Instrument("198.192.1.1")
s.timeout = 10
print("IDN:", s.ask("*IDN?"))

print("before SCALE :", s.ask("HORIZONTAL:SCALE?"))
print("before RECLEN:", s.ask("HORIZONTAL:RECORDLENGTH?"))
print("before XINCR :", s.ask("WFMOUTPRE:XINCR?"))
print("before NR_PT :", s.ask("WFMOUTPRE:NR_PT?"))

s.write("HORIZONTAL:SCALE 0.0001")
s.write("HORIZONTAL:RECORDLENGTH 100000")

print("after  SCALE :", s.ask("HORIZONTAL:SCALE?"))
print("after  RECLEN:", s.ask("HORIZONTAL:RECORDLENGTH?"))
print("after  XINCR :", s.ask("WFMOUTPRE:XINCR?"))
print("after  NR_PT :", s.ask("WFMOUTPRE:NR_PT?"))
print("ESR:", s.ask("*ESR?"))

s.close()
