import os
def read_txt(filename='signalDSID_table', txtdir='/nevis/katya01/data/users/kpark/SVJ/misc/'):
    d_m={}
    d_rinv={}
    if os.path.exists(txtdir+filename+'.txt'):
        with open(txtdir+filename+'.txt') as f:
            for i,line in enumerate(f):     
                line = line.split("/")
                filetag=line[0]
                line2=line[1].split(".")[-2].split("_") 
                rinv=line2[-1]
                m=line2[-2]
                d_m[filetag]=m
                d_rinv[filetag]=rinv 
    
    else:print('path does not exit', txtdir+filename+'.txt')
    return (d_m, d_rinv)

class Label:
    #proc_m, proc_rinv=SVJ_sep14.read_txt()
    proc_m, proc_rinv=read_txt()
#    print(proc_rinv)
#     proc_m={'QCDtest':np.nan,'515479': 500, '515482': 500, '515499':2000, '515502':2000, '515523':6000, '515526':6000}
#     proc_rinv={'QCDtest':np.nan, '515479': .2, '515482': .8, '515499':.2, '515502':.8, '515523':.2, '515526':.8}
    
    def __init__(self, name):
        self.name = name

    def get_string(self):
        string=self.name
        str_replace=['user','kipark','ebusch', 'mc20e', 'test','skim']
        if '.' in string:
            string=string.replace('.','')
        for rep in str_replace:
            if rep in self.name:
                string=string.replace(rep, '')
        return string

    def get_m(self, bool_num=False):
        string=Label.get_string(self)
        if 'output' in string or 'QCD' in string or 'Znunu' in string: return
        else:
            if bool_num: return int(Label.proc_m[string]) 
            else:return Label.proc_m[string] +' GeV'
    def get_rinv(self, bool_num=False):
        string=Label.get_string(self)
        if 'output' in string or 'QCD' in string or 'Znunu' in string: return        
        else:
            if bool_num: return int(Label.proc_rinv[string])/10
            else:return string(int(Label.proc_rinv[string])/10 )
    
    def get_label(self,bool_change=False):
        if not bool_change:
            string=Label.get_string(self)
            if 'output' in string or 'bkg' in string or 'QCD' in string or 'Znunu' in string: return f'{Label.get_string(self)}'.replace('test','')
            else:
                return f'{Label.proc_m[string]} GeV, {int(Label.proc_rinv[string])/10}'
        else:return self.name


