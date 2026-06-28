import sys, os
os.chdir(r'C:\Users\안가은\insicon-ai-mcs')
sys.path.insert(0, r'C:\Users\안가은\insicon-ai-mcs')

outpath = r'C:\Users\안가은\insicon-ai-mcs\experiments\dispatch_charging\eval_result.txt'
f = open(outpath, 'w', encoding='utf-8')

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

class Tee:
    def __init__(self, *streams): self._s = streams
    def write(self, x):
        for s in self._s: s.write(x)
    def flush(self):
        for s in self._s:
            try: s.flush()
            except: pass

sys.stdout = Tee(_orig_stdout, f)
sys.stderr = Tee(_orig_stderr, f)

try:
    from experiments.train_dispatch_charging import evaluate
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(
        r'C:\Users\안가은\insicon-ai-mcs\experiments\dispatch_charging\dc_n50_bs8_coupled_seed0_model'
    )
    evaluate(
        model,
        r'C:\Users\안가은\insicon-ai-mcs\experiments\dispatch_charging\dc_n50_bs8_coupled_seed0_vecnorm.pkl',
        n_amrs=50, max_steps=5000, battery_scale=8, n_ep=2, coupled=True, sim_days=7.0
    )
except Exception as e:
    print(f'ERROR: {e}', flush=True)
    import traceback; traceback.print_exc()
finally:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    f.close()
