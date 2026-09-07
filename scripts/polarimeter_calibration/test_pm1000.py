import Python_USB
import matlab

my_pm = Python_USB.initialize()

sel = matlab.uint16([3])          # USB 3.0 device
descr = "PM1000-100M-XL-FA-NN-D 44"

ok, diagnosis, handle, in_descr, out_descr = my_pm.initpm(
    sel, descr, nargout=5
)

res, ok = my_pm.readpm(matlab.double([512+45]), nargout=2)
print(res, ok)

my_pm.closepm()
my_pm.terminate()