import torch
ckpt = torch.load('checkpoints/gmf_refined_best.pt', map_location='cpu')
print("Keys:", ckpt.keys())
if 'aabb' in ckpt:
    print("AABB:", ckpt['aabb'])
elif 'bounds' in ckpt:
    print("Bounds:", ckpt['bounds'])
else:
    # Check if there is a config or metadata in the ckpt
    for k in ckpt.keys():
        if 'config' in k.lower() or 'meta' in k.lower():
            print(f"{k}:", ckpt[k])

# Check the range of means
m = ckpt['means']
print("Means shape:", m.shape)
print("Means range (min):", m.min(dim=0)[0])
print("Means range (max):", m.max(dim=0)[0])
