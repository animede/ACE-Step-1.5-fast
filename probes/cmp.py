import sys, glob, torch, torchaudio
def load(d):
    f = sorted(glob.glob(d + "/*.wav"))[0]; w, sr = torchaudio.load(f); return w.mean(0), sr, f
def logmel(w, sr):
    m = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=128)(w); return torch.log(m + 1e-6)
a, sr, fa = load(sys.argv[1])
for d in sys.argv[2:]:
    b, sr2, fb = load(d); n = min(len(a), len(b)); A, B = logmel(a[:n], sr), logmel(b[:n], sr)
    corr = torch.corrcoef(torch.stack([A.flatten(), B.flatten()]))[0,1].item()
    snr = 10*torch.log10(a[:n].pow(2).mean()/((a[:n]-b[:n]).pow(2).mean()+1e-12)).item()
    print(f"{d.split('/')[-1]:>18}: logmel_corr={corr:.4f} wave_snr={snr:.1f}dB  n={n/sr:.1f}s")
