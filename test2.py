import matplotlib.pyplot as plt
import numpy as np
def transform_loss(bkg_loss, sig_loss, make_plot=False, tag_file="", tag_title="", plot_dir='', bool_pfn=True):
    bkg_loss,sig_loss=equal_length(bkg_loss,sig_loss)
    nevents = len(sig_loss)

    truth_sig = np.ones(nevents)
    truth_bkg = np.zeros(nevents)
    truth_labels = np.concatenate((truth_bkg, truth_sig))
    eval_vals = np.concatenate((bkg_loss,sig_loss))
    eval_min = min(eval_vals)
    eval_max = max(eval_vals)-eval_min
    eval_transformed = [(x - eval_min)/eval_max for x in eval_vals]
    bkg_transformed = [(x - eval_min)/eval_max for x in bkg_loss]
    sig_transformed = [(x - eval_min)/eval_max for x in sig_loss]
    if make_plot:
        plot_score(bkg_transformed, sig_transformed, False, False, tag_file=tag_file+'_Transformed', tag_title=tag_title+'_Transformed', plot_dir=plot_dir, bool_pfn=True)
    return truth_labels, eval_vals

#transform_loss([1,2,3],[4,5,6])
import decimal
from decimal import Decimal
import sys
x = Decimal('0.1')
y = Decimal('0.1')
z = Decimal('0.1')

s = x + y + z

print(type(s),s, type(float(s)), float(s))
print(0.1+0.1+0.1)
sys.exit()


plt.plot([],[], label="Z'")
plt.legend()
plt.show()
