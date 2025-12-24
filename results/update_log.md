# Update Log — Density Computation Run

**Video:** D:\mini data 18-04\20250418_134655.mp4  
**Model:** D:\Major\models\uvh26.pt  
**Date:** <today's date>

## Steps Completed
- Ran detection + density computation
- Generated CSV outputs  
- Produced 15s window JSON files  
- Saved sample frames

## Observed Mislabels
- Frame XXXX: scooter → car (low confidence)
- Frame XXXX: autorickshaw → car
- Frame XXXX: truck → bus (conf 0.42)

## Density Samples
- results/samples/sample_0_xxxxx.jpg (lowest density)
- results/samples/sample_1_xxxxx.jpg (median density)
- results/samples/sample_2_xxxxx.jpg (highest density)

## Notes
- Density normalised using capacity = 60
- Video FPS = approx 30
