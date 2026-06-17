This dataset library is a standardized basic noise set for general purpose learning.
white, pink, blue, brown, violet, grey. 
sorted numerically.
all the files have a duration of 1 min, stereo, 32bit float bit depth.
the 1-6 are just standard noises with slight stereo decorrelation.
the 7-12 are the same but with L channel with inverted phase.
the 13-18 are the same but with R channel with inverted phase.
you could only just use the first 6 but for maximum coverage is recomended to use all the 3 sets at once, this will make the learning slower but more stable over time imho.
Theorically the first 6 are enough for UNlossifier but in order to give multiple view of the same problem, this approach is slower but more complete.