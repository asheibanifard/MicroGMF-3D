#!/bin/bash
# Re-export the PLY with the correct dimensions to fix Z-stretching
python3 export_ply.py --ckpt checkpoints/gmf_refined_best.pt --out viewer/scene.ply --dims 813,647,100
echo "Export complete. Scene saved to viewer/scene.ply"
