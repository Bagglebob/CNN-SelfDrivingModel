# DPS920_final_project

# Few Notes (Fawad):
1. I (Fawad) took a racing line and found that the model was terrible because of it.
    - essentially, hitting the apex of corners, hugging the inside of a corner, going from outside to inside
    - this resulted in the car trying to start the turn from the outside, but failed to give enough steering input to corner, resulting in understeer.
2. Additionally, I used keyboard to generate the training data at first. The caveats for this include:
    - Speed affects turn radius; I gathered data at a consistent 30km/h with the throttle down all the way.
    - At a lower speed, the steering inputs result in understeer (less distance is covered at 'x' steering, before the steering value goes back to 0)
    - I even tried to multiply the predicted steering by 1.5 but that didn't fix it.
3. Upon gathering data with **mouse steering** I found...